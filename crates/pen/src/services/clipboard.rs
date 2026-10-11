//! The system clipboard (arboard): copy text to it, or answer with the text
//! on it. It opens on first use and stays open, since on X11 whoever copied
//! serves the text until something else is copied. Without a clipboard (no
//! display, a Wayland session without XWayland) requests are answered
//! unavailable.
use pen_protocol::{clipboard_op, service};

use super::{Answer, Request, Service};

#[derive(Default)]
pub struct Clipboard {
    /// The clipboard, once opened, or why it could not be.
    opened: Option<Result<arboard::Clipboard, String>>,
}

impl Clipboard {
    fn open(&mut self) -> Option<&mut arboard::Clipboard> {
        let opened = self.opened.get_or_insert_with(|| {
            arboard::Clipboard::new().map_err(|error| {
                eprintln!("the clipboard is unavailable: {error}");
                error.to_string()
            })
        });
        opened.as_mut().ok()
    }
}

impl Service for Clipboard {
    fn code(&self) -> u32 {
        service::CLIPBOARD
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        let known = matches!(request.operation, clipboard_op::COPY | clipboard_op::PASTE);
        if !known {
            return Some(Answer::unknown());
        }
        let Some(clipboard) = self.open() else { return Some(Answer::unavailable()) };
        Some(match request.operation {
            clipboard_op::COPY => match clipboard.set_text(request.text()) {
                Ok(()) => Answer::ok(request.values.len() as f32),
                Err(error) => {
                    eprintln!("copying to the clipboard failed: {error}");
                    Answer::failed()
                }
            },
            _ => match clipboard.get_text() {
                Ok(text) => Answer::text(&text),
                // Nothing there, or nothing that is text.
                Err(_) => Answer::failed(),
            },
        })
    }
}
