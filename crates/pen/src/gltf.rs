//! `pen gltf FILE`: a glTF file imported as assets (pen-gltf), the work of an
//! asset recipe (gdev's Justfile runs it, and `pen dev` reruns that as the
//! file changes, pipeline.json saying so). It says what it wrote, and what
//! of the file it left out.
use std::path::Path;

use anyhow::Result;

pub fn import(source: &Path, out: &Path, options: &pen_gltf::Options) -> Result<()> {
    let imported = pen_gltf::import(source, out, options)?;
    for note in &imported.notes {
        eprintln!("gltf: {note}");
    }
    for key in &imported.removed {
        println!("removed {key}, which the file no longer makes");
    }
    for mesh in &imported.meshes {
        let skinned = if mesh.joints > 0 { format!(", {} joints", mesh.joints) } else { String::new() };
        let vat = mesh.vat.as_ref().map(|vat| format!(", baked into {vat}")).unwrap_or_default();
        println!("{} with {}{skinned}{vat}", mesh.mesh, mesh.material);
    }
    let clips: Vec<String> = imported
        .clips
        .iter()
        .enumerate()
        .map(|(index, clip)| format!("{index} {} ({:.2} s)", clip.name, clip.duration))
        .collect();
    if !clips.is_empty() {
        println!("clips: {}", clips.join(", "));
    }
    println!(
        "{}: {} meshes, {} materials, {} images into {}; {} files changed",
        source.display(),
        imported.meshes.len(),
        imported.materials.len(),
        imported.images.len(),
        out.display(),
        imported.changed.len()
    );
    Ok(())
}
