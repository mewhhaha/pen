//! Opens a web address in the player's browser: `xdg-open` on Linux (in the
//! Steam Linux Runtime and on a Steam Deck it hands the address to Steam),
//! `open` on macOS and the default handler on Windows. Only http and https
//! addresses, without spaces or control characters, are opened.
use std::process::{Command, Stdio};

use pen_protocol::{service, url_op};

use super::{Answer, Request, Service};

pub struct Url;

/// Whether `address` is a web address the browser may be given.
pub fn openable(address: &str) -> bool {
    (address.starts_with("https://") || address.starts_with("http://"))
        && address.len() <= 2048
        && address.chars().all(|char| !char.is_whitespace() && !char.is_control() && char != '"')
}

fn command(address: &str) -> Command {
    #[cfg(windows)]
    {
        let mut command = Command::new("rundll32.exe");
        command.args(["url.dll,FileProtocolHandler", address]);
        command
    }
    #[cfg(target_os = "macos")]
    {
        let mut command = Command::new("open");
        command.arg(address);
        command
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let mut command = Command::new("xdg-open");
        command.arg(address);
        command
    }
}

impl Service for Url {
    fn code(&self) -> u32 {
        service::URL
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        if request.operation != url_op::OPEN {
            return Some(Answer::unknown());
        }
        let address = request.text();
        if !openable(&address) {
            return Some(Answer::failed());
        }
        let spawned = command(&address).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn();
        Some(match spawned {
            Ok(mut child) => {
                // Reaped when it ends, without holding up the frame.
                std::thread::spawn(move || child.wait());
                Answer::ok(1.0)
            }
            Err(error) => {
                eprintln!("opening {address} failed: {error}");
                Answer::unavailable()
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_web_addresses_open() {
        assert!(openable("https://example.com/gdev?from=game"));
        assert!(openable("http://localhost:8000/?fresh"));
        assert!(!openable("file:///etc/passwd"));
        assert!(!openable("https://example.com/a b"));
        assert!(!openable("javascript:alert(1)"));
        assert!(!openable("https://example.com/\"&calc"));
    }
}
