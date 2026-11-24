//! Real HDRI skybox & IBL loader (Radiance `.hdr` RGBE + OpenEXR `.exr` scanlines)
//! with GPU equirectangular-to-GGX-prefiltered cubemap generation and L2 spherical
//! harmonic diffuse irradiance extraction.

use std::io::Read;

use anyhow::{Context, Result, bail, ensure};
use bytemuck::{Pod, Zeroable};
use flate2::read::ZlibDecoder;
use pen_protocol::Vector;

use crate::environment::{LEVELS as ENV_LEVELS, SIZE as ENV_SIZE};
use crate::pipeline_cache::PipelineCache;

const HDRI_IBL_WGSL: &str = include_str!("shaders/hdri_ibl.wgsl");

/// Default cubemap face resolution for prefiltered HDRI specular IBL.
pub const DEFAULT_HDRI_FACE_SIZE: u32 = 128;
/// Default number of GGX importance samples per rough mip texel.
pub const DEFAULT_HDRI_SAMPLES: u32 = 128;

/// Decoded linear floating-point panoramic equirectangular HDRI image (RGBA32F).
#[derive(Clone, Debug)]
pub struct HdriImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f32; 4]>,
}

impl HdriImage {
    /// Constructs an HDRI image directly from linear RGBA float32 pixels.
    pub fn from_rgba_f32(width: u32, height: u32, pixels: Vec<[f32; 4]>) -> Result<Self> {
        ensure!(width > 0 && height > 0, "HDRI dimensions must be positive");
        let expected = (width as usize).checked_mul(height as usize).context("HDRI dimensions overflow")?;
        ensure!(pixels.len() == expected, "HDRI pixel count mismatch: got {}, expected {expected}", pixels.len());
        let cleaned = pixels
            .into_iter()
            .map(|p| {
                [
                    if p[0].is_finite() { p[0].max(0.0) } else { 0.0 },
                    if p[1].is_finite() { p[1].max(0.0) } else { 0.0 },
                    if p[2].is_finite() { p[2].max(0.0) } else { 0.0 },
                    if p[3].is_finite() { p[3] } else { 1.0 },
                ]
            })
            .collect();
        Ok(Self { width, height, pixels: cleaned })
    }

    /// Auto-detects and decodes Radiance `.hdr` (RGBE) or OpenEXR `.exr` bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() >= 4 && bytes[0..4] == [0x76, 0x2f, 0x31, 0x01] {
            Self::from_exr_bytes(bytes)
        } else if bytes.starts_with(b"#?RADIANCE") || bytes.starts_with(b"#?RGBE") || bytes.starts_with(b"#?") {
            Self::from_hdr_bytes(bytes)
        } else {
            bail!("unrecognized HDRI format (expected Radiance .hdr or OpenEXR .exr)")
        }
    }

    /// Decodes Radiance `.hdr` (RGBE) files supporting flat and RLE scanlines.
    pub fn from_hdr_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = 0usize;
        let mut exposure = 1.0f32;
        let mut format_ok = false;

        // Parse header lines until blank line.
        while cursor < bytes.len() {
            let line_end = bytes[cursor..].iter().position(|&b| b == b'\n').map_or(bytes.len(), |p| cursor + p);
            let line_bytes = &bytes[cursor..line_end];
            cursor = (line_end + 1).min(bytes.len());
            let line = std::str::from_utf8(line_bytes).unwrap_or("").trim_end_matches('\r');
            if line.is_empty() {
                break;
            }
            if let Some(rest) = line.strip_prefix("FORMAT=") {
                if rest.trim() == "32-bit_rle_rgbe" {
                    format_ok = true;
                }
            } else if let Some(rest) = line.strip_prefix("EXPOSURE=")
                && let Ok(v) = rest.trim().parse::<f32>()
                && v.is_finite()
                && v > 0.0
            {
                exposure *= v;
            }
        }
        let _ = format_ok;

        // Parse resolution string (e.g. "-Y 512 +X 1024").
        let res_end = bytes[cursor..].iter().position(|&b| b == b'\n').map_or(bytes.len(), |p| cursor + p);
        let res_line = std::str::from_utf8(&bytes[cursor..res_end]).context("invalid HDR resolution line")?;
        cursor = (res_end + 1).min(bytes.len());
        let parts: Vec<&str> = res_line.split_whitespace().collect();
        ensure!(parts.len() == 4, "invalid HDR resolution specification: {res_line}");
        let flip_y = parts[0] == "+Y";
        ensure!(parts[0] == "-Y" || parts[0] == "+Y", "unsupported HDR vertical orientation: {}", parts[0]);
        ensure!(parts[2] == "+X" || parts[2] == "-X", "unsupported HDR horizontal orientation: {}", parts[2]);
        let flip_x = parts[2] == "-X";
        let height: u32 = parts[1].parse().context("invalid HDR height")?;
        let width: u32 = parts[3].parse().context("invalid HDR width")?;
        ensure!(width > 0 && height > 0 && width <= 16384 && height <= 8192, "HDR dimensions out of bounds");

        let w = width as usize;
        let h = height as usize;
        let mut pixels = vec![[0.0f32; 4]; w * h];
        let mut scanline = vec![0u8; w * 4];
        let inv_exposure = 1.0 / exposure.max(1e-6);

        for row in 0..h {
            if !(8..=0x7fff).contains(&w) {
                ensure!(cursor + w * 4 <= bytes.len(), "truncated flat HDR scanline");
                scanline.copy_from_slice(&bytes[cursor..cursor + w * 4]);
                cursor += w * 4;
            } else {
                ensure!(cursor + 4 <= bytes.len(), "truncated HDR scanline header");
                let b0 = bytes[cursor];
                let b1 = bytes[cursor + 1];
                let b2 = bytes[cursor + 2];
                let b3 = bytes[cursor + 3];
                if b0 == 2 && b1 == 2 && (b2 & 0x80) == 0 {
                    let encoded_width = ((b2 as usize) << 8) | (b3 as usize);
                    ensure!(encoded_width == w, "HDR RLE scanline width mismatch");
                    cursor += 4;
                    for channel in 0..4usize {
                        let mut x = 0usize;
                        while x < w {
                            ensure!(cursor < bytes.len(), "truncated HDR RLE stream");
                            let code = bytes[cursor];
                            cursor += 1;
                            if code > 128 {
                                let count = (code - 128) as usize;
                                ensure!(cursor < bytes.len() && x + count <= w, "invalid HDR RLE run");
                                let val = bytes[cursor];
                                cursor += 1;
                                for _ in 0..count {
                                    scanline[x * 4 + channel] = val;
                                    x += 1;
                                }
                            } else {
                                let count = code as usize;
                                ensure!(count > 0 && cursor + count <= bytes.len() && x + count <= w, "invalid HDR RLE literal");
                                for _ in 0..count {
                                    scanline[x * 4 + channel] = bytes[cursor];
                                    cursor += 1;
                                    x += 1;
                                }
                            }
                        }
                    }
                } else {
                    // Legacy flat scanline.
                    ensure!(cursor + w * 4 <= bytes.len(), "truncated legacy HDR scanline");
                    scanline.copy_from_slice(&bytes[cursor..cursor + w * 4]);
                    cursor += w * 4;
                }
            }

            let dst_y = if flip_y { h - 1 - row } else { row };
            for x in 0..w {
                let dst_x = if flip_x { w - 1 - x } else { x };
                let r = scanline[x * 4];
                let g = scanline[x * 4 + 1];
                let b = scanline[x * 4 + 2];
                let e = scanline[x * 4 + 3];
                let rgb = rgbe_to_rgb(r, g, b, e);
                pixels[dst_y * w + dst_x] = [
                    rgb[0] * inv_exposure,
                    rgb[1] * inv_exposure,
                    rgb[2] * inv_exposure,
                    1.0,
                ];
            }
        }
        Ok(Self { width, height, pixels })
    }

    /// Decodes single-part scanline OpenEXR `.exr` files (NONE / RLE / ZIPS / ZIP; HALF & FLOAT channels).
    pub fn from_exr_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() >= 8 && bytes[0..4] == [0x76, 0x2f, 0x31, 0x01], "not an OpenEXR file");
        let version_flags = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let version = version_flags & 0xff;
        ensure!(version == 2, "unsupported OpenEXR version {version}");
        ensure!((version_flags & 0x200) == 0, "tiled OpenEXR is not supported; export scanline EXR");
        ensure!((version_flags & 0x800) == 0 && (version_flags & 0x1000) == 0, "deep/multipart OpenEXR unsupported");

        let mut cursor = 8usize;
        #[derive(Clone, Debug)]
        struct ExrChannel {
            name: String,
            pixel_type: u32, // 1 = HALF, 2 = FLOAT
        }
        let mut channels: Vec<ExrChannel> = Vec::new();
        let mut compression: u8 = 0;
        let mut data_window: Option<[i32; 4]> = None;

        // Header attributes terminated by an empty NUL name.
        loop {
            ensure!(cursor < bytes.len(), "unexpected EOF in EXR header");
            if bytes[cursor] == 0 {
                cursor += 1;
                break;
            }
            let attr_name = read_cstr(bytes, &mut cursor)?;
            let _attr_type = read_cstr(bytes, &mut cursor)?;
            let attr_size = read_u32_le(bytes, &mut cursor)? as usize;
            ensure!(cursor + attr_size <= bytes.len(), "truncated EXR attribute {attr_name}");
            let attr_data = &bytes[cursor..cursor + attr_size];
            cursor += attr_size;

            match attr_name.as_str() {
                "channels" => {
                    let mut c = 0usize;
                    while c < attr_data.len() {
                        if attr_data[c] == 0 {
                            break;
                        }
                        let ch_name = read_cstr(attr_data, &mut c)?;
                        ensure!(c + 16 <= attr_data.len(), "truncated EXR channel list");
                        let pixel_type = u32::from_le_bytes(attr_data[c..c + 4].try_into().unwrap());
                        c += 16; // pixel_type(4), pLinear(1), reserved(3), xSampling(4), ySampling(4)
                        ensure!(pixel_type == 1 || pixel_type == 2, "unsupported EXR channel type {pixel_type}");
                        channels.push(ExrChannel { name: ch_name, pixel_type });
                    }
                }
                "compression" => {
                    ensure!(!attr_data.is_empty(), "empty EXR compression attribute");
                    compression = attr_data[0];
                }
                "dataWindow" => {
                    ensure!(attr_data.len() == 16, "invalid EXR dataWindow");
                    data_window = Some([
                        i32::from_le_bytes(attr_data[0..4].try_into().unwrap()),
                        i32::from_le_bytes(attr_data[4..8].try_into().unwrap()),
                        i32::from_le_bytes(attr_data[8..12].try_into().unwrap()),
                        i32::from_le_bytes(attr_data[12..16].try_into().unwrap()),
                    ]);
                }
                _ => {}
            }
        }

        let [x_min, y_min, x_max, y_max] = data_window.context("EXR missing dataWindow")?;
        ensure!(x_max >= x_min && y_max >= y_min, "invalid EXR dataWindow");
        let width = u32::try_from(i64::from(x_max) - i64::from(x_min) + 1).context("EXR width overflow")?;
        let height = u32::try_from(i64::from(y_max) - i64::from(y_min) + 1).context("EXR height overflow")?;
        ensure!(width > 0 && height > 0 && width <= 16384 && height <= 8192, "EXR dimensions out of bounds");
        ensure!(!channels.is_empty(), "EXR missing channels");

        let lines_per_block: usize = match compression {
            0..=2 => 1, // NO_COMPRESSION, RLE, ZIPS
            3 => 16,        // ZIP
            other => bail!("unsupported EXR compression mode {other} (supported: NONE, RLE, ZIPS, ZIP)"),
        };
        let num_blocks = (height as usize).div_ceil(lines_per_block);
        let mut offsets = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            ensure!(cursor + 8 <= bytes.len(), "truncated EXR offset table");
            let off = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            offsets.push(usize::try_from(off).context("EXR offset overflow")?);
        }

        let w = width as usize;
        let h = height as usize;
        let bytes_per_pixel: usize = channels.iter().map(|c| if c.pixel_type == 1 { 2 } else { 4 }).sum();
        let mut pixels = vec![[0.0f32, 0.0, 0.0, 1.0]; w * h];

        for &block_off in &offsets {
            ensure!(block_off + 8 <= bytes.len(), "EXR block offset out of bounds");
            let y_coord = i32::from_le_bytes(bytes[block_off..block_off + 4].try_into().unwrap());
            let packed_size = u32::from_le_bytes(bytes[block_off + 4..block_off + 8].try_into().unwrap()) as usize;
            let data_start = block_off + 8;
            ensure!(data_start + packed_size <= bytes.len(), "truncated EXR scanline block");
            let packed = &bytes[data_start..data_start + packed_size];

            let start_row = usize::try_from(i64::from(y_coord) - i64::from(y_min)).context("EXR row out of bounds")?;
            let block_lines = (h - start_row).min(lines_per_block);
            let raw_size = block_lines * w * bytes_per_pixel;

            let raw = match compression {
                0 => {
                    ensure!(packed.len() == raw_size, "uncompressed EXR block size mismatch");
                    packed.to_vec()
                }
                1 => {
                    let uncomp = if packed.len() < raw_size {
                        exr_rle_decode(packed, raw_size)?
                    } else {
                        packed.to_vec()
                    };
                    exr_unpredict_and_reorder(&uncomp)
                }
                2 | 3 => {
                    let uncomp = if packed.len() < raw_size {
                        let mut decoder = ZlibDecoder::new(packed);
                        let mut buf = Vec::with_capacity(raw_size);
                        decoder.read_to_end(&mut buf).context("EXR zlib decompression failed")?;
                        ensure!(buf.len() == raw_size, "EXR zlib decompressed size mismatch");
                        buf
                    } else {
                        packed.to_vec()
                    };
                    exr_unpredict_and_reorder(&uncomp)
                }
                _ => unreachable!(),
            };

            // Channels within each scanline are stored planar in alphabetical header order.
            let mut pos = 0usize;
            for line_idx in 0..block_lines {
                let row = start_row + line_idx;
                for ch in &channels {
                    let slot = match ch.name.as_str() {
                        "R" | "r" => Some(0usize),
                        "G" | "g" => Some(1usize),
                        "B" | "b" => Some(2usize),
                        "A" | "a" => Some(3usize),
                        "Y" | "y" => Some(4usize), // luminance
                        _ => None,
                    };
                    let stride = if ch.pixel_type == 1 { 2 } else { 4 };
                    ensure!(pos + w * stride <= raw.len(), "EXR scanline channel overrun");
                    for x in 0..w {
                        let val = if ch.pixel_type == 1 {
                            f16_to_f32(u16::from_le_bytes([raw[pos + x * 2], raw[pos + x * 2 + 1]]))
                        } else {
                            f32::from_le_bytes(raw[pos + x * 4..pos + x * 4 + 4].try_into().unwrap())
                        };
                        let clean = if val.is_finite() { val } else { 0.0 };
                        match slot {
                            Some(0..=2) => pixels[row * w + x][slot.unwrap()] = clean.max(0.0),
                            Some(3) => pixels[row * w + x][3] = clean,
                            Some(4) => {
                                let c = clean.max(0.0);
                                pixels[row * w + x][0] = c;
                                pixels[row * w + x][1] = c;
                                pixels[row * w + x][2] = c;
                            }
                            _ => {}
                        }
                    }
                    pos += w * stride;
                }
            }
        }
        Ok(Self { width, height, pixels })
    }

    /// Encodes this HDRI as an uncompressed scanline OpenEXR `.exr` (F32 RGBA) — useful for round-trip tools & tests.
    pub fn to_exr_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&[0x76, 0x2f, 0x31, 0x01]);
        out.extend_from_slice(&2u32.to_le_bytes());

        // channels attribute (alphabetical: A, B, G, R)
        out.extend_from_slice(b"channels\0chlist\0");
        let mut chlist = Vec::new();
        for name in ["A", "B", "G", "R"] {
            chlist.extend_from_slice(name.as_bytes());
            chlist.push(0);
            chlist.extend_from_slice(&2u32.to_le_bytes()); // FLOAT
            chlist.extend_from_slice(&[0, 0, 0, 0]);
            chlist.extend_from_slice(&1i32.to_le_bytes());
            chlist.extend_from_slice(&1i32.to_le_bytes());
        }
        chlist.push(0);
        out.extend_from_slice(&u32::try_from(chlist.len()).expect("EXR header fits u32").to_le_bytes());
        out.extend_from_slice(&chlist);

        // compression = 0 (NONE)
        out.extend_from_slice(b"compression\0compression\0");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.push(0);

        // dataWindow
        out.extend_from_slice(b"dataWindow\0box2i\0");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&0i32.to_le_bytes());
        out.extend_from_slice(&0i32.to_le_bytes());
        #[allow(clippy::cast_possible_wrap, reason = "EXR dimensions bounded")]
        out.extend_from_slice(&((self.width as i32) - 1).to_le_bytes());
        #[allow(clippy::cast_possible_wrap, reason = "EXR dimensions bounded")]
        out.extend_from_slice(&((self.height as i32) - 1).to_le_bytes());

        // End of header
        out.push(0);

        let w = self.width as usize;
        let h = self.height as usize;
        let line_payload = w * 16; // 4 channels * f32
        let offset_table_start = out.len();
        let data_start = offset_table_start + h * 8;
        for row in 0..h {
            let off = (data_start + row * (8 + line_payload)) as u64;
            out.extend_from_slice(&off.to_le_bytes());
        }
        for row in 0..h {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "EXR row index")]
            out.extend_from_slice(&(row as i32).to_le_bytes());
            #[allow(clippy::cast_possible_truncation, reason = "EXR scanline bytes")]
            out.extend_from_slice(&(line_payload as u32).to_le_bytes());
            for ch in [3usize, 2, 1, 0] {
                for x in 0..w {
                    out.extend_from_slice(&self.pixels[row * w + x][ch].to_le_bytes());
                }
            }
        }
        out
    }

    /// Projects the equirectangular HDRI onto 9 L2 real spherical harmonics
    /// matching Pen's `environment.wgsl` (`sky_ambient`: cosine lobe applied and
    /// divided by pi, basis order `1, y, z, x, xy, yz, 3zz-1, xz, xx-yy`).
    pub fn project_sh_l2(&self, intensity: f32, rotation_rad: f32) -> [Vector; 9] {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut sh = [[0.0f64; 3]; 9];
        let mut total_solid_angle = 0.0f64;
        let d_phi = std::f64::consts::TAU / w as f64;
        let d_theta = std::f64::consts::PI / h as f64;
        let rot = f64::from(rotation_rad);
        let gain = f64::from(intensity.max(0.0));

        // Cosine lobe transfer weights (divided by PI so sky_ambient evaluates irradiance / pi).
        // A0 = pi, A1 = 2*pi/3, A2 = pi/4 -> divided by pi: 1.0, 2/3, 1/4.
        let band_scale = [
            1.0f64,
            2.0 / 3.0,
            2.0 / 3.0,
            2.0 / 3.0,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
        ];

        for y in 0..h {
            let theta = (y as f64 + 0.5) * d_theta;
            let sin_t = theta.sin();
            let cos_t = theta.cos();
            let dw = sin_t * d_theta * d_phi;
            for x in 0..w {
                let u = (x as f64 + 0.5) / w as f64;
                // Inverse of u = phi / 2pi + 0.5 with yaw rotation applied.
                let phi = (u - 0.5) * std::f64::consts::TAU - rot;
                let dx = sin_t * phi.cos();
                let dy = cos_t;
                let dz = sin_t * phi.sin();

                let basis = [
                    0.282_094_791_8,
                    0.488_602_511_9 * dy,
                    0.488_602_511_9 * dz,
                    0.488_602_511_9 * dx,
                    1.092_548_430_6 * dx * dy,
                    1.092_548_430_6 * dy * dz,
                    0.315_391_565_3 * (3.0 * dz * dz - 1.0),
                    1.092_548_430_6 * dx * dz,
                    0.546_274_215_3 * (dx * dx - dy * dy),
                ];

                let p = self.pixels[y * w + x];
                let rgb = [f64::from(p[0]) * gain, f64::from(p[1]) * gain, f64::from(p[2]) * gain];
                for i in 0..9 {
                    let wt = basis[i] * dw;
                    sh[i][0] += rgb[0] * wt;
                    sh[i][1] += rgb[1] * wt;
                    sh[i][2] += rgb[2] * wt;
                }
                total_solid_angle += dw;
            }
        }

        let norm = if total_solid_angle > 1e-12 {
            (4.0 * std::f64::consts::PI) / total_solid_angle
        } else {
            1.0
        };
        #[allow(clippy::cast_possible_truncation, reason = "SH coefficients are f32 GPU uniforms")]
        std::array::from_fn(|i| {
            let s = norm * band_scale[i];
            Vector { x: (sh[i][0] * s) as f32, y: (sh[i][1] * s) as f32, z: (sh[i][2] * s) as f32 }
        })
    }
}

/// Linearly blends two L2 SH diffuse irradiance sets (e.g. indoor/outdoor transitions).
pub fn blend_sh_l2(a: &[Vector; 9], b: &[Vector; 9], t: f32) -> [Vector; 9] {
    let w = t.clamp(0.0, 1.0);
    let inv = 1.0 - w;
    std::array::from_fn(|i| Vector {
        x: a[i].x * inv + b[i].x * w,
        y: a[i].y * inv + b[i].y * w,
        z: a[i].z * inv + b[i].z * w,
    })
}

fn rgbe_to_rgb(r: u8, g: u8, b: u8, e: u8) -> [f32; 3] {
    if e == 0 {
        return [0.0; 3];
    }
    let scale = (2.0f32).powi(i32::from(e) - 128 - 8);
    [f32::from(r) * scale, f32::from(g) * scale, f32::from(b) * scale]
}

fn read_cstr(bytes: &[u8], cursor: &mut usize) -> Result<String> {
    let end = bytes[*cursor..].iter().position(|&b| b == 0).context("unterminated EXR C string")?;
    let s = std::str::from_utf8(&bytes[*cursor..*cursor + end]).context("invalid UTF-8 in EXR header")?.to_owned();
    *cursor += end + 1;
    Ok(s)
}

fn read_u32_le(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    ensure!(*cursor + 4 <= bytes.len(), "unexpected EOF reading EXR u32");
    let v = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x3ff);
    let f_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal half -> normal f32.
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(f_bits)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "IEEE 754 binary16 bit packing"
)]
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x007f_ffff;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> (1 - exp);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    if exp >= 31 {
        return sign | 0x7bff; // clamp to max finite f16
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn exr_rle_decode(src: &[u8], expected: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(expected);
    let mut i = 0usize;
    while i < src.len() {
        let count = src[i].cast_signed();
        i += 1;
        if count < 0 {
            let n = usize::from(count.unsigned_abs());
            ensure!(i + n <= src.len() && out.len() + n <= expected, "invalid EXR RLE literal run");
            out.extend_from_slice(&src[i..i + n]);
            i += n;
        } else {
            let n = usize::from(count.cast_unsigned()) + 1;
            ensure!(i < src.len() && out.len() + n <= expected, "invalid EXR RLE repeat run");
            let val = src[i];
            i += 1;
            out.resize(out.len() + n, val);
        }
    }
    ensure!(out.len() == expected, "EXR RLE decoded length mismatch");
    Ok(out)
}

fn exr_unpredict_and_reorder(src: &[u8]) -> Vec<u8> {
    let mut pred = src.to_vec();
    for i in 1..pred.len() {
        let d = i32::from(pred[i - 1]) + i32::from(pred[i]) - 128;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "EXR predictor mod 256")]
        {
            pred[i] = (d & 0xff) as u8;
        }
    }
    let len = pred.len();
    let half = len.div_ceil(2);
    let mut out = vec![0u8; len];
    for i in 0..len {
        out[i] = if i % 2 == 0 { pred[i / 2] } else { pred[half + i / 2] };
    }
    out
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrefilterParams {
    face: u32,
    size: u32,
    samples: u32,
    _pad: u32,
    roughness: f32,
    intensity: f32,
    rotation_rad: f32,
    _pad2: f32,
}

/// GPU prefiltered cubemap generated from an equirectangular HDRI image.
pub struct HdriPrefilterer {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

/// Result of prefiltering an HDRI onto a GPU cubemap ready for `Environment`.
pub struct HdriCubemap {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sh_l2: [Vector; 9],
    pub face_size: u32,
    pub mip_levels: u32,
}

impl HdriPrefilterer {
    pub fn new(device: &wgpu::Device, cache: Option<&wgpu::PipelineCache>) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hdri_ibl"),
            source: wgpu::ShaderSource::Wgsl(HDRI_IBL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hdri_ibl_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<PrefilterParams>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hdri_ibl_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hdri_ibl_prefilter"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("prefilter_main"),
            compilation_options: Default::default(),
            cache,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hdri_equirect_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self { pipeline, layout, sampler }
    }

    /// Uploads `image` as `Rgba16Float`, prefilters it into a GGX specular cubemap
    /// (`face_size` x `face_size` x 6 with `mip_levels` mips), and returns the cubemap + L2 SH.
    #[allow(clippy::too_many_arguments, reason = "explicit GPU cubemap prefilter parameters")]
    pub fn build_cubemap(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        image: &HdriImage,
        face_size: u32,
        mip_levels: u32,
        samples: u32,
        intensity: f32,
        rotation_rad: f32,
    ) -> Result<HdriCubemap> {
        ensure!(face_size >= 8 && face_size.is_power_of_two(), "HDRI cubemap face_size must be power of two >= 8");
        let max_mips = face_size.trailing_zeros() + 1;
        let mips = mip_levels.clamp(1, max_mips);

        // Convert RGBA32F -> RGBA16F for universal filterable texture support on WebGPU & Vulkan.
        let mut f16_data = Vec::with_capacity(image.pixels.len() * 4);
        for p in &image.pixels {
            f16_data.push(f32_to_f16(p[0]));
            f16_data.push(f32_to_f16(p[1]));
            f16_data.push(f32_to_f16(p[2]));
            f16_data.push(f32_to_f16(p[3]));
        }
        let equirect = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdri_equirect"),
            size: wgpu::Extent3d { width: image.width, height: image.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            equirect.as_image_copy(),
            bytemuck::cast_slice(&f16_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(image.width * 8),
                rows_per_image: Some(image.height),
            },
            wgpu::Extent3d { width: image.width, height: image.height, depth_or_array_layers: 1 },
        );
        let equirect_view = equirect.create_view(&Default::default());

        let cube = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdri_sky_cube"),
            size: wgpu::Extent3d { width: face_size, height: face_size, depth_or_array_layers: 6 },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let total_dispatches = (mips as usize) * 6;
        let stride = device.limits().min_uniform_buffer_offset_alignment.max(256) as usize;
        let mut uniform_bytes = vec![0u8; total_dispatches * stride];
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdri_prefilter_uniforms"),
            size: uniform_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut dispatch_idx = 0usize;
        for mip in 0..mips {
            let mip_size = (face_size >> mip).max(1);
            let roughness = if mips > 1 { mip as f32 / (mips - 1) as f32 } else { 0.0 };
            let mip_samples = if mip == 0 { 1 } else { samples.max(16) };
            for face in 0..6u32 {
                let p = PrefilterParams {
                    face,
                    size: mip_size,
                    samples: mip_samples,
                    _pad: 0,
                    roughness,
                    intensity: intensity.max(0.0),
                    rotation_rad,
                    _pad2: 0.0,
                };
                let off = dispatch_idx * stride;
                uniform_bytes[off..off + std::mem::size_of::<PrefilterParams>()]
                    .copy_from_slice(bytemuck::bytes_of(&p));
                dispatch_idx += 1;
            }
        }
        queue.write_buffer(&uniform_buf, 0, &uniform_bytes);

        let mut groups = Vec::with_capacity(total_dispatches);
        dispatch_idx = 0;
        for mip in 0..mips {
            for face in 0..6u32 {
                let face_view = cube.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("hdri_face_mip"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: face,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let off = (dispatch_idx * stride) as u64;
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hdri_prefilter_bg"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &uniform_buf,
                                offset: off,
                                size: wgpu::BufferSize::new(std::mem::size_of::<PrefilterParams>() as u64),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&equirect_view),
                        },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&face_view),
                        },
                    ],
                });
                groups.push((mip, bg));
                dispatch_idx += 1;
            }
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hdri_prefilter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            for (mip, bg) in &groups {
                let mip_size = (face_size >> *mip).max(1);
                let wg = mip_size.div_ceil(8);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(wg, wg, 1);
            }
        }

        let view = cube.create_view(&wgpu::TextureViewDescriptor {
            label: Some("hdri_sky_cube_view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let sh_l2 = image.project_sh_l2(intensity, rotation_rad);
        Ok(HdriCubemap { texture: cube, view, sh_l2, face_size, mip_levels: mips })
    }

    /// Convenience helper matching `Environment`'s native cubemap contract (`ENV_SIZE`, `ENV_LEVELS`).
    pub(crate) fn build_for_environment(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cache: &PipelineCache,
        image: &HdriImage,
        intensity: f32,
        rotation_rad: f32,
    ) -> Result<HdriCubemap> {
        let prefilterer = Self::new(device, cache.get());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("hdri_env") });
        let face_size = DEFAULT_HDRI_FACE_SIZE.max(ENV_SIZE);
        let cube = prefilterer.build_cubemap(
            device,
            queue,
            &mut encoder,
            image,
            face_size,
            ENV_LEVELS,
            DEFAULT_HDRI_SAMPLES,
            intensity,
            rotation_rad,
        )?;
        queue.submit([encoder.finish()]);
        Ok(cube)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radiance_hdr_and_exr_round_trip_and_sh_projection() {
        // Synthetic bright-top / dark-bottom HDRI: verify EXR round-trip, HDR RLE decode, and L2 SH.
        let (w, h) = (32u32, 16u32);
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            let up = y < h / 2;
            let rgb = if up { [4.0f32, 2.0, 1.0, 1.0] } else { [0.2, 0.2, 0.4, 1.0] };
            for _ in 0..w {
                pixels.push(rgb);
            }
        }
        let img = HdriImage::from_rgba_f32(w, h, pixels).unwrap();
        let exr = img.to_exr_bytes();
        let decoded = HdriImage::from_bytes(&exr).unwrap();
        assert_eq!(decoded.width, w);
        assert_eq!(decoded.height, h);
        assert!((decoded.pixels[0][0] - 4.0).abs() < 1e-5);
        assert!((decoded.pixels[(h as usize - 1) * w as usize][2] - 0.4).abs() < 1e-5);

        // Construct a valid flat Radiance .hdr file and decode it.
        let mut hdr = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n".to_vec();
        // 4x2 = 8 pixels of [2.0, 1.0, 0.5] -> exponent 130 (2^2 = 4), mantissas [128, 64, 32]
        for _ in 0..8 {
            hdr.extend_from_slice(&[128u8, 64, 32, 130]);
        }
        let hdr_img = HdriImage::from_bytes(&hdr).unwrap();
        assert_eq!((hdr_img.width, hdr_img.height), (4, 2));
        assert!((hdr_img.pixels[0][0] - 2.0).abs() < 0.02);
        assert!((hdr_img.pixels[0][1] - 1.0).abs() < 0.02);
        assert!((hdr_img.pixels[0][2] - 0.5).abs() < 0.02);

        // Constant white HDRI -> SH band 0 evaluates to (1,1,1) for any normal.
        let white = HdriImage::from_rgba_f32(16, 8, vec![[1.0, 1.0, 1.0, 1.0]; 128]).unwrap();
        let sh = white.project_sh_l2(1.0, 0.0);
        let eval_up = sh[0].x * 0.282_094_79 + sh[1].x * 0.488_602_51;
        assert!((eval_up - 1.0).abs() < 0.03, "constant white HDRI SH eval was {eval_up}");
    }

    #[test]
    fn gpu_hdri_prefilter_produces_directional_hdr_cubemap() {
        let Some((device, queue)) = crate::testing::device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        // Upper hemisphere bright warm [8, 4, 2], lower hemisphere cool [0.2, 0.4, 1.0].
        let (w, h) = (32u32, 16u32);
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            let c = if y < h / 2 { [8.0f32, 4.0, 2.0, 1.0] } else { [0.2, 0.4, 1.0, 1.0] };
            for _ in 0..w {
                pixels.push(c);
            }
        }
        let img = HdriImage::from_rgba_f32(w, h, pixels).unwrap();
        let prefilterer = HdriPrefilterer::new(&device, None);
        let mut encoder = device.create_command_encoder(&Default::default());
        let cube = prefilterer
            .build_cubemap(&device, &queue, &mut encoder, &img, 16, 4, 32, 1.0, 0.0)
            .unwrap();

        // Read back +Y face (layer 2) and -Y face (layer 3) at mip 0 (16x16 Rgba16Float -> 8 bytes/px).
        let padded_row = 256u32;
        let read_face = |encoder: &mut wgpu::CommandEncoder, layer: u32| -> wgpu::Buffer {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hdri_readback"),
                size: u64::from(padded_row * 16),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &cube.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(16),
                    },
                },
                wgpu::Extent3d { width: 16, height: 16, depth_or_array_layers: 1 },
            );
            buf
        };
        let up_buf = read_face(&mut encoder, 2);
        let down_buf = read_face(&mut encoder, 3);
        queue.submit([encoder.finish()]);

        let sample_center = |buf: &wgpu::Buffer| -> [f32; 3] {
            let slice = buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            rx.recv().unwrap().unwrap();
            let mapped = slice.get_mapped_range().unwrap();
            let base = 8 * padded_row as usize + 8 * 8;
            let r = f16_to_f32(u16::from_le_bytes([mapped[base], mapped[base + 1]]));
            let g = f16_to_f32(u16::from_le_bytes([mapped[base + 2], mapped[base + 3]]));
            let b = f16_to_f32(u16::from_le_bytes([mapped[base + 4], mapped[base + 5]]));
            drop(mapped);
            buf.unmap();
            [r, g, b]
        };

        let up = sample_center(&up_buf);
        let down = sample_center(&down_buf);
        assert!(up[0] > 7.0 && up[0] > up[2] * 3.0, "+Y face should capture warm HDR sky, got {up:?}");
        assert!(down[2] > 0.8 && down[2] > down[0] * 3.0, "-Y face should capture cool ground, got {down:?}");
    }
}
