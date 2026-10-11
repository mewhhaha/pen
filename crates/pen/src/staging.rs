//! Build complete outputs beside their destination, then publish them.
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};

pub struct Staging {
    destination: PathBuf,
    pub directory: PathBuf,
}

/// Resolve symlinked existing prefixes, including an output not there yet.
fn resolved(path: &Path) -> Result<PathBuf> {
    let mut absolute = PathBuf::new();
    for component in std::path::absolute(path)?.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                absolute.pop();
            }
            component => absolute.push(component.as_os_str()),
        }
    }
    let mut prefix = absolute.as_path();
    while !prefix.exists() {
        prefix = prefix.parent().context("output has no existing parent")?;
    }
    Ok(prefix.canonicalize()?.join(absolute.strip_prefix(prefix)?))
}

impl Staging {
    pub fn new(root: &Path, destination: &Path) -> Result<Self> {
        let root = root.canonicalize()?;
        let destination = resolved(destination)?;
        ensure!(
            !root.starts_with(&destination),
            "output {} contains the checkout",
            destination.display()
        );
        ensure!(!destination.is_file(), "output must be a directory");
        // Allow only build directories inside this project or another checkout.
        // New source directories are protected without maintaining an inventory.
        for checkout in std::iter::once(root.as_path()).chain(
            destination
                .ancestors()
                .filter(|path| path.join(".git").exists()),
        ) {
            if let Ok(relative) = destination.strip_prefix(checkout) {
                let first = relative.components().next();
                ensure!(
                    matches!(first, Some(Component::Normal(name)) if name == ".pen" || name == "dist" || name == "target"),
                    "output {} overlaps checkout sources; use .pen, dist, target or a directory outside the checkout",
                    destination.display()
                );
            }
        }
        let parent = destination.parent().context("output has no parent")?;
        std::fs::create_dir_all(parent)?;
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let name = destination
            .file_name()
            .context("output has no filename")?
            .to_string_lossy();
        let directory = parent.join(format!(".{name}.build-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&directory)?;
        Ok(Self {
            destination,
            directory,
        })
    }

    pub fn publish(self) -> Result<()> {
        let mut backup_name = self.directory.as_os_str().to_os_string();
        backup_name.push(".previous");
        let backup = PathBuf::from(backup_name);
        let previous = self.destination.exists();
        if previous {
            std::fs::rename(&self.destination, &backup)
                .context("preserving the previous output")?;
        }
        if let Err(error) = std::fs::rename(&self.directory, &self.destination) {
            if previous {
                std::fs::rename(&backup, &self.destination)
                    .context("restoring the previous output")?;
            }
            return Err(error).context("publishing the completed output");
        }
        if previous && let Err(error) = std::fs::remove_dir_all(&backup) {
            eprintln!(
                "published {}; old output remains at {}: {error}",
                self.destination.display(),
                backup.display()
            );
        }
        Ok(())
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_build_preserves_output_and_success_replaces_it() {
        let root = crate::project::temporary_dir()
            .unwrap()
            .join(format!("staging-{}", std::process::id()));
        std::fs::create_dir_all(root.join("src")).unwrap();
        let output = root.join(".pen/package");
        std::fs::create_dir_all(&output).unwrap();
        std::fs::write(output.join("old"), b"last good build").unwrap();
        assert!(Staging::new(&root, &root).is_err());
        for source in [
            "src/package",
            "assets",
            "scripts",
            "test_support",
            "future-inputs",
        ] {
            assert!(Staging::new(&root, &root.join(source)).is_err());
        }
        let stage = Staging::new(&root, &output).unwrap();
        std::fs::write(stage.directory.join("partial"), b"incomplete").unwrap();
        drop(stage);
        assert_eq!(
            std::fs::read(output.join("old")).unwrap(),
            b"last good build"
        );
        let stage = Staging::new(&root, &output).unwrap();
        std::fs::write(stage.directory.join("new"), b"complete").unwrap();
        stage.publish().unwrap();
        assert!(!output.join("old").exists());
        assert_eq!(std::fs::read(output.join("new")).unwrap(), b"complete");
        std::fs::remove_dir_all(root).unwrap();
    }
}
