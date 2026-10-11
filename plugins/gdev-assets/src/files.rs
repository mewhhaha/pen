//! The checkout's files as the plugin reads them, noting what the result
//! depends on: each file read, each directory listed, and for each path that
//! is missing, the nearest directory above it that is there, whose new files
//! would bring it.
use std::cell::RefCell;
use std::collections::BTreeSet;

use crate::{Files, paths};

pub(crate) struct Tracked<'a> {
    files: &'a dyn Files,
    /// The checkout's root: a missing path's nearest directory is never above
    /// it.
    root: String,
    dependencies: RefCell<BTreeSet<String>>,
}

impl<'a> Tracked<'a> {
    pub(crate) fn new(files: &'a dyn Files, root: String) -> Self {
        Self { files, root, dependencies: RefCell::default() }
    }

    pub(crate) fn read(&self, path: &str) -> Result<Vec<u8>, String> {
        let read = self.files.read(path);
        self.depend(path, read.is_ok());
        read
    }

    pub(crate) fn read_directory(&self, directory: &str) -> Result<Vec<String>, String> {
        let names = self.files.read_directory(directory);
        self.depend(directory, names.is_ok());
        names
    }

    pub(crate) fn exists(&self, path: &str) -> bool {
        let exists = self.files.exists(path);
        self.depend(path, exists);
        exists
    }

    /// What was noted, sorted.
    pub(crate) fn dependencies(self) -> Vec<String> {
        self.dependencies.into_inner().into_iter().collect()
    }

    /// Notes `path` when it is `there`, else where it would appear.
    fn depend(&self, path: &str, there: bool) {
        if there { self.note(path) } else { self.missing(path) }
    }

    fn note(&self, path: &str) {
        self.dependencies.borrow_mut().insert(path.to_string());
    }

    /// Notes the nearest directory above `path` that is there, up to the
    /// root.
    fn missing(&self, path: &str) {
        let mut directory = paths::dirname(path);
        loop {
            if self.files.exists(&directory) {
                return self.note(&directory);
            }
            if directory == self.root || directory == "/" {
                return;
            }
            directory = paths::dirname(&directory);
        }
    }
}
