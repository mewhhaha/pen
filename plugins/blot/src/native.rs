//! A [`Child`] that is a process of this machine's, for running the client
//! natively (its tests) as pen's `process.child` runs it in the component.
use std::io::{Read, Write};
use std::path::Path;
use std::process::{ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::{Child, ReadError};

/// How much of its standard error a child keeps.
const ERROR_TAIL: usize = 64 * 1024;

/// A running process with piped standard input and output. Its output is
/// read on a thread, so reads can time out; its standard error is kept on
/// another. Dropping it kills it.
pub struct NativeChild {
    process: std::process::Child,
    input: Option<ChildStdin>,
    /// Chunks of its output; disconnected at its end.
    output: Receiver<Result<Vec<u8>, String>>,
    /// What came of a chunk and was not read yet.
    pending: Vec<u8>,
    errors: Arc<Mutex<Vec<u8>>>,
    /// Reading its standard error, until it exits.
    error_reader: Option<JoinHandle<()>>,
}

impl NativeChild {
    /// Starts `program` with `args` in `directory`, its environment only
    /// `env`.
    pub fn spawn(
        program: &Path,
        args: &[String],
        env: &[(String, String)],
        directory: &Path,
    ) -> std::io::Result<NativeChild> {
        let mut process = Command::new(program)
            .args(args)
            .env_clear()
            .envs(env.iter().map(|(name, value)| (name, value)))
            .current_dir(directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let (sender, output) = channel();
        let mut stdout = process.stdout.take().expect("piped");
        std::thread::spawn(move || {
            let mut buffer = vec![0; 64 * 1024];
            loop {
                let chunk = match stdout.read(&mut buffer) {
                    Ok(0) => return,
                    Ok(count) => Ok(buffer[..count].to_vec()),
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(error) => Err(error.to_string()),
                };
                let failed = chunk.is_err();
                if sender.send(chunk).is_err() || failed {
                    return;
                }
            }
        });
        let errors = Arc::new(Mutex::new(Vec::new()));
        let mut stderr = process.stderr.take().expect("piped");
        let kept = errors.clone();
        let error_reader = std::thread::spawn(move || {
            let mut buffer = vec![0; 16 * 1024];
            while let Ok(count @ 1..) = stderr.read(&mut buffer) {
                let mut kept = kept.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                kept.extend_from_slice(&buffer[..count]);
                let excess = kept.len().saturating_sub(ERROR_TAIL);
                kept.drain(..excess);
            }
        });
        let input = process.stdin.take();
        Ok(NativeChild { process, input, output, pending: Vec::new(), errors, error_reader: Some(error_reader) })
    }

    /// Its process id.
    pub fn id(&self) -> u32 {
        self.process.id()
    }

    /// The exit code once it exited, after the rest of its standard error.
    fn exited(&mut self, status: std::process::ExitStatus) -> Option<i32> {
        if let Some(reader) = self.error_reader.take() {
            let _ = reader.join();
        }
        status.code()
    }
}

impl Child for NativeChild {
    fn write(&mut self, bytes: &[u8]) -> Result<(), String> {
        let input = self.input.as_mut().ok_or("its input is closed")?;
        input.write_all(bytes).and_then(|()| input.flush()).map_err(|error| error.to_string())
    }

    fn close_input(&mut self) {
        self.input = None;
    }

    fn read(&mut self, max: u32, timeout_ms: Option<u32>) -> Result<Vec<u8>, ReadError> {
        if self.pending.is_empty() {
            let chunk = match timeout_ms {
                None => self.output.recv().map_err(|_| ReadError::Closed),
                Some(ms) => self.output.recv_timeout(Duration::from_millis(ms.into())).map_err(|error| match error {
                    RecvTimeoutError::Timeout => ReadError::Timeout,
                    RecvTimeoutError::Disconnected => ReadError::Closed,
                }),
            };
            self.pending = chunk?.map_err(ReadError::Other)?;
        }
        let rest = self.pending.split_off(self.pending.len().min(max.max(1) as usize));
        Ok(std::mem::replace(&mut self.pending, rest))
    }

    fn error_tail(&mut self) -> String {
        String::from_utf8_lossy(&self.errors.lock().unwrap_or_else(|poisoned| poisoned.into_inner())).into_owned()
    }

    fn wait(&mut self, timeout_ms: Option<u32>) -> Option<i32> {
        let Some(ms) = timeout_ms else {
            let status = self.process.wait().ok()?;
            return self.exited(status);
        };
        let deadline = Instant::now() + Duration::from_millis(ms.into());
        loop {
            match self.process.try_wait() {
                Ok(Some(status)) => return self.exited(status),
                Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(5)),
                _ => return None,
            }
        }
    }

    fn kill(&mut self) {
        let _ = self.process.kill();
    }
}

impl Drop for NativeChild {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}
