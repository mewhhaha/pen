//! Bootstrap protocol generation without compiling consumers of the new schema.
use anyhow::{Context, Result, bail};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut project = None;
    let mut rust = None;
    let mut check = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--project" => project = Some(PathBuf::from(args.next().context("--project needs a directory")?)),
            "--rust" => rust = Some(PathBuf::from(args.next().context("--rust needs a path")?)),
            "--check" => check = true,
            _ => bail!("usage: pen-codegen --project DIR [--rust FILE] [--check]"),
        }
    }
    pen_codegen::generate(&project.context("--project is required")?, rust.as_deref(), check)
}
