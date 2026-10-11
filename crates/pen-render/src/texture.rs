// PNG and JPEG textures, decoded to RGBA8 pixels (texture.ts). In the browser
// the page's own decoder does this (platform.rs).
pub(crate) struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn decode_texture(bytes: &[u8]) -> anyhow::Result<DecodedImage> {
    let format = if bytes.starts_with(&[137, 80, 78, 71]) {
        image::ImageFormat::Png
    } else if bytes.starts_with(&[255, 216]) {
        image::ImageFormat::Jpeg
    } else {
        anyhow::bail!("texture must be PNG or JPEG");
    };
    // Grey, RGB and CMYK pixels all widen to opaque RGBA, as texture.ts does.
    let decoded = image::load_from_memory_with_format(bytes, format)?.into_rgba8();
    Ok(DecodedImage { width: decoded.width(), height: decoded.height(), pixels: decoded.into_raw() })
}
