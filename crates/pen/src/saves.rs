//! The player's own saves in the data directory, the quick save and the
//! edited scene: written stamped (`snapshot::Stamp`: when, and the saves
//! each replaced), with the last [`BACKUPS`] versions kept in `backups/`,
//! and checked when the game starts against what this device last wrote.
//!
//! Steam Auto-Cloud syncs those files between the player's devices
//! (docs/shipping.md). When two devices each saved since they last synced,
//! Steam asks the player which to keep, and the other is gone from the
//! cloud. Not from here: `backups/` is never synced, backup 1 is always what
//! this device last wrote or found, and [`check`] puts a save that came
//! from elsewhere among the backups before anything can overwrite it. It
//! tells, by the save's lineage, whether it continued from this device's
//! last save (the ordinary case), is an older one of this device's (a copy
//! that lagged, or one restored by hand), or forked from an older one on
//! another device (a conflict Steam resolved for it), and says so. The game
//! plays the save it finds: the player, or Steam for them, chose it.
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use crate::devlog::{self, Source};
use crate::snapshot::{self, LINEAGE, SavedState, Stamp, decode_stamped, encode_stamped, identity};

/// Versions of each save kept in `backups/`: this device's last write is the
/// first, then those before it, and saves that came from elsewhere.
pub const BACKUPS: usize = 5;

/// Backup `number` (1 the newest) of `path`: `backups/quicksave.1.bin`
/// beside `quicksave.bin`.
pub fn backup(path: &Path, number: usize) -> PathBuf {
    let directory = path.parent().unwrap_or(Path::new(".")).join("backups");
    let stem = path.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_default();
    match path.extension() {
        Some(extension) => directory.join(format!("{stem}.{number}.{}", extension.to_string_lossy())),
        None => directory.join(format!("{stem}.{number}")),
    }
}

fn read(path: &Path) -> Result<Option<Vec<u8>>> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}

/// Shifts the backups of `path` down one, the oldest going, and keeps
/// `bytes` as backup 1.
fn keep(path: &Path, bytes: &[u8]) -> Result<()> {
    let oldest = backup(path, BACKUPS);
    if oldest.exists() {
        std::fs::remove_file(&oldest).with_context(|| format!("deleting {}", oldest.display()))?;
    }
    for number in (1..BACKUPS).rev() {
        let from = backup(path, number);
        if from.exists() {
            std::fs::rename(&from, backup(path, number + 1)).with_context(|| format!("moving {}", from.display()))?;
        }
    }
    snapshot::replace(&backup(path, 1), bytes)
}

fn now() -> u64 {
    let since = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
}

/// Writes `saved` to `path` as this device's newest save, stamped with the
/// save it replaces, then keeps it as backup 1. A save there that is not
/// among the backups yet (one that came from elsewhere since [`check`]) is
/// kept first.
pub fn write(path: &Path, saved: &SavedState) -> Result<()> {
    let current = read(path)?;
    let latest = read(&backup(path, 1))?;
    if let Some(current) = &current
        && latest.as_deref().map(identity) != Some(identity(current))
    {
        keep(path, current)?;
    }
    let lineage = match &current {
        Some(bytes) => {
            let before = decode_stamped(bytes).ok().and_then(|(_, stamp)| stamp).unwrap_or_default();
            std::iter::once(identity(bytes)).chain(before.lineage).take(LINEAGE).collect()
        }
        None => Vec::new(),
    };
    let bytes = encode_stamped(saved, &Stamp { written: now(), lineage })?;
    // The save first: a crash before its backup leaves a save that
    // continued from backup 1, which `check` takes in as such.
    snapshot::replace(path, &bytes)?;
    keep(path, &bytes)
}

/// What [`check`] found of a save at startup, beside this device's last.
#[derive(Clone, Debug, PartialEq)]
pub enum Found {
    /// No save there.
    Nothing,
    /// This device's last save, or a save with nothing here to compare with.
    Own,
    /// Another device's, which continued from this device's last save.
    Continued,
    /// An older save than this device's last, from its own past.
    Older,
    /// Another device's, written at `written`, which went on from an older
    /// save than this device's last: both played on since they last synced.
    Forked { written: u64 },
    /// A save without a stamp (an older pen's, a copy, or not a save at all).
    Unstamped,
}

impl Found {
    /// What to tell the player, when it is not the ordinary case: where
    /// this device's own last save is kept.
    pub fn warning(&self, path: &Path) -> Option<String> {
        let name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        let kept = backup(path, 2);
        let kept = kept.display();
        match self {
            Found::Nothing | Found::Own | Found::Continued => None,
            Found::Older => Some(format!(
                "{name} is older than the one this device last saved, which is kept as {kept}; the older one plays"
            )),
            Found::Forked { written } => Some(format!(
                "{name} was saved on another device ({}) since this one last synced, and this device's own last \
                 save, which differs, is kept as {kept}",
                crate::crash::timestamp(written / 1000)
            )),
            Found::Unstamped => {
                Some(format!("{name} was replaced by an unstamped save; this device's last save is kept as {kept}"))
            }
        }
    }
}

/// At startup, before the save is read: tells the save at `path` from this
/// device's last write (backup 1), and when it differs keeps it as backup 1
/// too, so that the next save overwrites neither.
pub fn check(path: &Path) -> Result<Found> {
    let Some(bytes) = read(path)? else { return Ok(Found::Nothing) };
    let Some(latest) = read(&backup(path, 1))? else {
        keep(path, &bytes)?;
        return Ok(Found::Own);
    };
    let id = identity(&bytes);
    if identity(&latest) == id {
        return Ok(Found::Own);
    }
    let stamp = |bytes: &[u8]| decode_stamped(bytes).ok().and_then(|(_, stamp)| stamp);
    let found = match stamp(&bytes) {
        None => Found::Unstamped,
        Some(found) if found.lineage.contains(&identity(&latest)) => Found::Continued,
        Some(found) => {
            let mut past = stamp(&latest).map(|stamp| stamp.lineage).unwrap_or_default();
            for number in 2..=BACKUPS {
                past.extend(read(&backup(path, number))?.as_deref().map(identity));
            }
            if past.contains(&id) { Found::Older } else { Found::Forked { written: found.written } }
        }
    };
    keep(path, &bytes)?;
    Ok(found)
}

/// [`check`]s each of the player's saves as a session starts, and says
/// what is not the ordinary case.
pub fn check_all<'a>(paths: impl IntoIterator<Item = &'a PathBuf>) {
    for path in paths {
        match check(path) {
            Ok(found) => {
                if let Some(warning) = found.warning(path) {
                    devlog::error(Source::Session, warning);
                }
            }
            Err(error) => devlog::error(Source::Session, format!("checking {}: {error:#}", path.display())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn directory(name: &str) -> PathBuf {
        let directory = crate::project::temporary_dir().unwrap().join(format!("saves-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    fn state(value: f32) -> SavedState {
        SavedState { schema: 3, state: vec![value; 4] }
    }

    fn value(path: &Path) -> f32 {
        snapshot::read(path).unwrap().state[0]
    }

    #[test]
    fn saves_keep_their_backups_and_lineage() {
        let root = directory("backups");
        let path = root.join("quicksave.bin");
        assert_eq!(backup(&path, 2), root.join("backups/quicksave.2.bin"));
        assert_eq!(backup(&root.join("save"), 1), root.join("backups/save.1"));
        assert_eq!(check(&path).unwrap(), Found::Nothing);
        for number in 1..=7 {
            write(&path, &state(number as f32)).unwrap();
            assert_eq!(check(&path).unwrap(), Found::Own);
        }
        assert_eq!(value(&path), 7.0);
        // The newest BACKUPS, the save itself the first.
        let kept: Vec<f32> = (1..=BACKUPS).map(|number| value(&backup(&path, number))).collect();
        assert_eq!(kept, [7.0, 6.0, 5.0, 4.0, 3.0]);
        assert!(!backup(&path, BACKUPS + 1).exists());
        let (_, stamp) = decode_stamped(&std::fs::read(&path).unwrap()).unwrap();
        let stamp = stamp.unwrap();
        assert_eq!(stamp.lineage.len(), 6);
        assert_eq!(stamp.lineage[0], identity(&std::fs::read(backup(&path, 2)).unwrap()));
        assert!(stamp.written > 1_700_000_000_000);
        std::fs::remove_dir_all(root).unwrap();
    }

    /// Two devices that share the save, as Steam Cloud has them: each
    /// directory is one device's data directory, and `sync` copies the save
    /// from one to the other as Steam does.
    #[test]
    fn saves_from_other_devices_are_told_apart_and_kept() {
        let root = directory("cloud");
        let (deck, desktop) = (root.join("deck/quicksave.bin"), root.join("desktop/quicksave.bin"));
        let sync = |from: &Path, to: &Path| {
            std::fs::create_dir_all(to.parent().unwrap()).unwrap();
            std::fs::copy(from, to).unwrap();
        };
        write(&deck, &state(1.0)).unwrap();
        sync(&deck, &desktop);
        // The desktop never saw it: nothing to compare with, so its own.
        assert_eq!(check(&desktop).unwrap(), Found::Own);
        write(&desktop, &state(2.0)).unwrap();
        sync(&desktop, &deck);
        // The deck finds the desktop's, which went on from its own.
        assert_eq!(check(&deck).unwrap(), Found::Continued);
        assert_eq!(Found::Continued.warning(&deck), None);
        assert_eq!(value(&backup(&deck, 1)), 2.0);
        assert_eq!(value(&backup(&deck, 2)), 1.0);

        // Both play on offline from 2, and the cloud keeps the desktop's.
        write(&deck, &state(3.0)).unwrap();
        write(&desktop, &state(4.0)).unwrap();
        sync(&desktop, &deck);
        let found = check(&deck).unwrap();
        assert!(matches!(found, Found::Forked { written } if written > 1_700_000_000_000), "{found:?}");
        let warning = found.warning(&deck).unwrap();
        assert!(warning.contains("another device") && warning.contains("quicksave.2.bin"), "{warning}");
        // The desktop's plays; the deck's own is backup 2, not lost to the
        // next save.
        assert_eq!(value(&deck), 4.0);
        assert_eq!(value(&backup(&deck, 2)), 3.0);
        write(&deck, &state(5.0)).unwrap();
        let kept: Vec<f32> = (1..=4).map(|number| value(&backup(&deck, number))).collect();
        assert_eq!(kept, [5.0, 4.0, 3.0, 2.0]);

        // An older save of the deck's own comes back (restored by hand).
        std::fs::copy(backup(&deck, 3), &deck).unwrap();
        assert_eq!(check(&deck).unwrap(), Found::Older);
        assert_eq!(value(&deck), 3.0);
        assert_eq!(value(&backup(&deck, 2)), 5.0);

        // An unstamped one (an older pen's) is kept too.
        snapshot::write(&deck, &state(6.0)).unwrap();
        assert_eq!(check(&deck).unwrap(), Found::Unstamped);
        assert_eq!(value(&backup(&deck, 2)), 3.0);
        // A save written over one check never saw keeps it first.
        snapshot::write(&deck, &state(7.0)).unwrap();
        write(&deck, &state(8.0)).unwrap();
        assert_eq!((value(&backup(&deck, 1)), value(&backup(&deck, 2))), (8.0, 7.0));
        std::fs::remove_dir_all(root).unwrap();
    }
}
