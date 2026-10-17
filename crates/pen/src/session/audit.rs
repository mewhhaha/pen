//! The accessibility audit's frames (`pen_runtime::audit`): the game's
//! interface and its pixels as one frame drew them, without the dev tools,
//! read back when the dev tools ask (DEVTOOLS_OP_AUDIT, which answers what
//! the last frame read back shows and has the next one read back) or, with
//! PEN_A11Y_AUDIT=FILE, whenever the interface changes, its findings then
//! written to FILE when they change, in a window or headless.
use std::fs::File;
use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use pen_protocol::UiNode;
use pen_runtime::audit::{self, Finding, Picture};

use super::gpu::Gpu;

/// A frame's interface and its pixels (RGBA).
struct Captured {
    ui: Vec<UiNode>,
    size: (u32, u32),
    rgba: Vec<u8>,
}

#[derive(Default)]
pub(super) struct Audits {
    /// The dev tools asked: the next frame is read back.
    wanted: bool,
    last: Option<Captured>,
    /// PEN_A11Y_AUDIT's file, and the findings last written to it.
    file: Option<(PathBuf, File)>,
    written: Option<Vec<Finding>>,
    /// Frames drawn, to say which frame a finding is from.
    frames: u64,
}

impl Audits {
    pub fn from_env() -> Result<Audits> {
        let Some(path) = std::env::var_os("PEN_A11Y_AUDIT").filter(|path| !path.is_empty()).map(PathBuf::from) else {
            return Ok(Audits::default());
        };
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(&path).with_context(|| format!("writing PEN_A11Y_AUDIT {}", path.display()))?;
        Ok(Audits { file: Some((path, file)), ..Audits::default() })
    }

    /// Whether the frame about to draw, declaring `ui`, is read back.
    pub fn due(&mut self, ui: &[UiNode]) -> bool {
        self.frames += 1;
        self.wanted || (self.file.is_some() && self.last.as_ref().is_none_or(|last| last.ui != ui))
    }

    /// Reads the frame just drawn into `texture` back, with its interface.
    pub fn capture(&mut self, gpu: &Gpu, texture: &wgpu::Texture, ui: &[UiNode]) -> Result<()> {
        let mut rgba = gpu.read(texture)?;
        if gpu.bgra() {
            for pixel in rgba.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
        }
        self.wanted = false;
        self.last = Some(Captured { ui: ui.to_vec(), size: (texture.width(), texture.height()), rgba });
        if self.file.is_some() {
            let findings = self.findings(ui);
            self.write(findings);
        }
        Ok(())
    }

    /// What the audit finds in the last frame read back, or in `current`
    /// (the game's interface now) without its pixels before one is.
    pub fn findings(&self, current: &[UiNode]) -> Vec<Finding> {
        match &self.last {
            Some(captured) => {
                let picture = Picture { width: captured.size.0, height: captured.size.1, rgba: &captured.rgba };
                audit::audit(&captured.ui, Some(&picture))
            }
            None => audit::audit(current, None),
        }
    }

    /// The dev tools ask: what the last frame shows, and the next read back.
    #[cfg(feature = "devtools")]
    pub fn ask(&mut self, current: &[UiNode]) -> Vec<Finding> {
        self.wanted = true;
        self.findings(current)
    }

    fn write(&mut self, findings: Vec<Finding>) {
        if self.written.as_ref() == Some(&findings) {
            return;
        }
        if let Some((path, file)) = &mut self.file {
            let text = format!("frame {}\n{}\n", self.frames.saturating_sub(1), audit::report(&findings));
            if let Err(error) = file.write_all(text.as_bytes()) {
                eprintln!("PEN_A11Y_AUDIT: writing {}: {error}", path.display());
                self.file = None;
            }
        }
        self.written = Some(findings);
    }
}
