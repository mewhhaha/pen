//! The component: wit/pen-plugin.wit's `compiler` over [`crate::Session`],
//! whose compiler runs as pen's `process.child`, and its `tests` over
//! [`crate::find_tests`], which reads through pen's `files`.
use std::cell::RefCell;

use crate::{Build, Expectation, Overlay, Program};

wit_bindgen::generate!({ path: "../../wit", world: "compiler-plugin" });

use exports::pen::plugin::compiler::{self, BuildResult, GuestSession};
use exports::pen::plugin::tests;
use pen::plugin::log::{self, Level};
use pen::plugin::{files, process, types};

struct Plugin;

export!(Plugin);

impl compiler::Guest for Plugin {
    type Session = Compiler;
}

/// The checkout's files, through what pen.json grants the plugin.
struct Granted;

impl crate::Files for Granted {
    fn read(&mut self, path: &str) -> Result<Vec<u8>, String> {
        files::read(path)
    }

    fn read_directory(&mut self, directory: &str) -> Result<Vec<String>, String> {
        files::read_directory(directory)
    }

    fn is_file(&mut self, path: &str) -> bool {
        files::is_file(path)
    }
}

impl tests::Guest for Plugin {
    fn find(
        program: types::Program,
        paths: Vec<String>,
        filter: Option<String>,
    ) -> Result<Vec<tests::TestModule>, String> {
        let program = Program { root: program.root, entry: program.entry, settings: program.settings };
        let modules = crate::find_tests(&program, &paths, filter.as_deref(), &mut Granted)?;
        Ok(modules
            .into_iter()
            .map(|module| tests::TestModule {
                path: module.path,
                entry: module.entry,
                overlays: module
                    .overlays
                    .into_iter()
                    .map(|overlay| types::Overlay { path: overlay.path, contents: overlay.contents })
                    .collect(),
                prefix: module.prefix,
                expect: match module.expect {
                    Expectation::Checks => tests::Expectation::Checks,
                    Expectation::Traps => tests::Expectation::Traps,
                    Expectation::Rejected(text) => tests::Expectation::Rejected(text),
                },
            })
            .collect())
    }
}

/// pen's child process, as the client's.
struct Process(process::Child);

impl crate::Child for Process {
    fn write(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.0.write(bytes)
    }

    fn close_input(&mut self) {
        self.0.close_input();
    }

    fn read(&mut self, max: u32, timeout_ms: Option<u32>) -> Result<Vec<u8>, crate::ReadError> {
        self.0.read(max, timeout_ms).map_err(|error| match error {
            process::ReadError::Closed => crate::ReadError::Closed,
            process::ReadError::Timeout => crate::ReadError::Timeout,
            process::ReadError::Other(error) => crate::ReadError::Other(error),
        })
    }

    fn error_tail(&mut self) -> String {
        self.0.error_tail()
    }

    fn wait(&mut self, timeout_ms: Option<u32>) -> Option<i32> {
        self.0.wait(timeout_ms)
    }

    fn kill(&mut self) {
        self.0.kill();
    }
}

/// A session: pen calls it through shared references.
struct Compiler(RefCell<crate::Session<Process>>);

impl GuestSession for Compiler {
    fn open(program: types::Program) -> Result<compiler::Session, String> {
        let program = Program { root: program.root, entry: program.entry, settings: program.settings };
        let session = crate::Session::open(&program, &process::cache_directory(), |name, args, env| {
            process::spawn(name, args, env).map(Process)
        })?;
        log::write(Level::Info, &format!("Blot compiler {} opened {}", session.identity(), session.entry()));
        Ok(compiler::Session::new(Compiler(RefCell::new(session))))
    }

    fn build(&self, overlays: Vec<types::Overlay>) -> Result<BuildResult, String> {
        let overlays: Vec<Overlay> =
            overlays.into_iter().map(|overlay| Overlay { path: overlay.path, contents: overlay.contents }).collect();
        Ok(match self.0.borrow_mut().build(&overlays)? {
            Build::Built(wasm) => BuildResult::Built(wasm),
            Build::Rejected(diagnostics) => BuildResult::Rejected(
                diagnostics
                    .into_iter()
                    .map(|diagnostic| types::Diagnostic {
                        file: diagnostic.file,
                        stage: diagnostic.stage,
                        code: diagnostic.code,
                        start: diagnostic.start,
                        end: diagnostic.end,
                        message: diagnostic.message,
                        details: diagnostic.details,
                    })
                    .collect(),
            ),
        })
    }

    fn watched(&self) -> Vec<String> {
        self.0.borrow().watched().to_vec()
    }

    fn close(&self) -> Result<(), String> {
        self.0.borrow_mut().close()
    }
}
