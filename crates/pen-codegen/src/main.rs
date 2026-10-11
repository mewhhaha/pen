use anyhow::{Result, bail};
use std::path::PathBuf;
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut project = PathBuf::from(".");
    let mut check = false;
    let mut rust = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--project" => {
                project = args
                    .next()
                    .map(PathBuf::from)
                    .ok_or_else(|| anyhow::anyhow!("--project needs a path"))?
            }
            "--rust" => {
                rust = Some(
                    args.next()
                        .map(PathBuf::from)
                        .ok_or_else(|| anyhow::anyhow!("--rust needs a path"))?,
                )
            }
            "--check" => check = true,
            "--help" | "-h" => {
                println!("pen-codegen [--project PATH] [--rust PATH] [--check]");
                return Ok(());
            }
            _ => bail!("unknown argument {arg}"),
        }
    }
    pen_codegen::generate(&project, rust.as_deref(), check)
}
