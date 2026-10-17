//! Bundles Gather as pen plays a game: `bundle GAME.wasm OUT` writes
//! `OUT/game.wasm` (the module `adapt` gives the guest ABI's shape),
//! `OUT/assets` and `OUT/game.json`. `pen --game OUT` plays the bundle, and
//! `pen build --game OUT` packages it with pen's player runtime.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    use anyhow::{Context, bail};
    use std::path::Path;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let [wasm, out] = &args[..] else { bail!("usage: bundle GAME.wasm OUT") };
    let raw = std::fs::read(wasm).with_context(|| format!("reading {wasm}"))?;
    let adapted = rust_guest::adapt::adapt(&raw).with_context(|| format!("adapting {wasm}"))?;
    let out = Path::new(out);
    std::fs::create_dir_all(out.join("assets"))?;
    std::fs::write(out.join("game.wasm"), &adapted)?;
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
    for entry in std::fs::read_dir(&assets)? {
        let entry = entry?;
        std::fs::copy(entry.path(), out.join("assets").join(entry.file_name()))?;
    }
    std::fs::write(out.join("game.json"), "{\n  \"name\": \"gather\",\n  \"title\": \"Gather\"\n}\n")?;
    eprintln!("bundled {} ({} bytes of Wasm, {} as built)", out.display(), adapted.len(), raw.len());
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
