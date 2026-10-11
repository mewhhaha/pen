//! The component: wit/pen-plugin.wit's `compiler` over [`crate::Session`],
//! whose compiler runs as pen's `process.child`.
use std::cell::RefCell;

use crate::{Build, Overlay, Program};

wit_bindgen::generate!({ path: "../../wit", world: "compiler-plugin" });

use exports::pen::plugin::compiler::{self, BuildResult, GuestSession};
use pen::plugin::log::{self, Level};
use pen::plugin::{process, types};

struct Plugin;

export!(Plugin);

impl compiler::Guest for Plugin {
    type Session = Compiler;
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
                    })
                    .collect(),
            ),
        })
    }

    fn watched(&self) -> Vec<String> {
        self.0.borrow().watched().to_vec()
    }
}

impl Drop for Compiler {
    /// Closes the compiler, which saves its restart cache, so that the next
    /// session starts from this one's work.
    fn drop(&mut self) {
        if let Err(error) = self.0.get_mut().close() {
            log::write(Level::Error, &format!("closing the Blot compiler: {error}"));
        }
    }
}
