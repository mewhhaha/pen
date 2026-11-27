//! GPU planar YUV420p video-to-texture playback: validates planar Y/U/V frames,
//! demuxes Y4M (`YUV4MPEG2`) uncompressed streams and IVF (`DKIF`) compressed
//! frame containers, manages wall-clock playback timing, and converts YUV420p
//! planes directly into an `Rgba8Unorm` GPU texture (with an `Rgba8UnormSrgb`
//! view so scene materials sample display-referred video in linear light) via
//! `yuv420_to_rgba.wgsl`.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "pixel & timestamp math")]

use anyhow::{Context, Result, bail, ensure};

/// Maximum single video dimension accepted by GPU upload and container parsers.
pub const MAX_VIDEO_DIM: u32 = 8192;

/// YCbCr color matrix standard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMatrix {
    Bt709,
    Bt601,
}

/// YCbCr quantization range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorRange {
    Limited,
    Full,
}

/// 4:2:0 chroma sample siting relative to the 2x2 luma block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ChromaSiting {
    /// Center of the 2x2 luma block (JPEG / `C420jpeg`).
    Centered = 0,
    /// Vertically centered, horizontally co-sited with the left luma column (MPEG-2 / H.264 / BT.709).
    Left = 1,
    /// Co-sited with the top-left luma sample (PAL-DV / `C420paldv`).
    TopLeft = 2,
}

/// YCbCr color matrix and quantization range combined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ColorSpace {
    Bt709Limited = 0,
    Bt709Full = 1,
    Bt601Limited = 2,
    Bt601Full = 3,
}

impl ColorSpace {
    pub fn from_parts(matrix: ColorMatrix, range: ColorRange) -> Self {
        match (matrix, range) {
            (ColorMatrix::Bt709, ColorRange::Limited) => Self::Bt709Limited,
            (ColorMatrix::Bt709, ColorRange::Full) => Self::Bt709Full,
            (ColorMatrix::Bt601, ColorRange::Limited) => Self::Bt601Limited,
            (ColorMatrix::Bt601, ColorRange::Full) => Self::Bt601Full,
        }
    }
}

/// Validated planar YUV420p video frame (even dimensions, 4:2:0 half-res chroma).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Yuv420Frame {
    pub width: u32,
    pub height: u32,
    pub color_space: ColorSpace,
    pub siting: ChromaSiting,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

impl Yuv420Frame {
    pub fn new(width: u32, height: u32, color_space: ColorSpace, y: Vec<u8>, u: Vec<u8>, v: Vec<u8>) -> Result<Self> {
        Self::with_siting(width, height, color_space, ChromaSiting::Left, y, u, v)
    }

    pub fn with_siting(
        width: u32,
        height: u32,
        color_space: ColorSpace,
        siting: ChromaSiting,
        y: Vec<u8>,
        u: Vec<u8>,
        v: Vec<u8>,
    ) -> Result<Self> {
        let frame = Self { width, height, color_space, siting, y, u, v };
        frame.validate(MAX_VIDEO_DIM)?;
        Ok(frame)
    }

    /// Re-validates dimensions, `max_dim` device cap, and exact plane lengths before GPU upload.
    pub fn validate(&self, max_dim: u32) -> Result<()> {
        let limit = max_dim.min(MAX_VIDEO_DIM);
        ensure!(
            self.width >= 2 && self.height >= 2 && self.width <= limit && self.height <= limit,
            "invalid video dimensions {}x{} (limit {limit})",
            self.width,
            self.height
        );
        ensure!(
            self.width.is_multiple_of(2) && self.height.is_multiple_of(2),
            "YUV420p requires even dimensions, got {}x{}",
            self.width,
            self.height
        );
        let y_len = (self.width as usize).checked_mul(self.height as usize).context("video plane overflow")?;
        let uv_len =
            ((self.width / 2) as usize).checked_mul((self.height / 2) as usize).context("video chroma overflow")?;
        ensure!(self.y.len() == y_len, "Y plane size {} != expected {y_len}", self.y.len());
        ensure!(self.u.len() == uv_len, "U plane size {} != expected {uv_len}", self.u.len());
        ensure!(self.v.len() == uv_len, "V plane size {} != expected {uv_len}", self.v.len());
        Ok(())
    }
}

/// Encodes an sRGB 8-bit RGB triple into BT.709 or BT.601 YCbCr bytes (for tests and tooling).
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8, color_space: ColorSpace) -> (u8, u8, u8) {
    let (rf, gf, bf) = (f64::from(r) / 255.0, f64::from(g) / 255.0, f64::from(b) / 255.0);
    let (kr, kb) = match color_space {
        ColorSpace::Bt709Limited | ColorSpace::Bt709Full => (0.2126, 0.0722),
        ColorSpace::Bt601Limited | ColorSpace::Bt601Full => (0.299, 0.114),
    };
    let kg = 1.0 - kr - kb;
    let y = kr * rf + kg * gf + kb * bf;
    let cb = 0.5 * (bf - y) / (1.0 - kb);
    let cr = 0.5 * (rf - y) / (1.0 - kr);
    match color_space {
        ColorSpace::Bt709Limited | ColorSpace::Bt601Limited => (
            (16.0 + 219.0 * y).round().clamp(0.0, 255.0) as u8,
            (128.0 + 224.0 * cb).round().clamp(0.0, 255.0) as u8,
            (128.0 + 224.0 * cr).round().clamp(0.0, 255.0) as u8,
        ),
        ColorSpace::Bt709Full | ColorSpace::Bt601Full => (
            (255.0 * y).round().clamp(0.0, 255.0) as u8,
            (128.0 + 255.0 * cb).round().clamp(0.0, 255.0) as u8,
            (128.0 + 255.0 * cr).round().clamp(0.0, 255.0) as u8,
        ),
    }
}

/// Parsed `YUV4MPEG2` (`.y4m`) uncompressed planar YUV420p stream.
#[derive(Clone, Debug)]
pub struct Y4mStream {
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub color_space: ColorSpace,
    pub siting: ChromaSiting,
    pub frames: Vec<Yuv420Frame>,
}

impl Y4mStream {
    pub fn fps(&self) -> f64 {
        if self.fps_den == 0 || self.fps_num == 0 { 30.0 } else { f64::from(self.fps_num) / f64::from(self.fps_den) }
    }

    /// Parses a Y4M stream defaulting to BT.709 matrix and Limited range unless
    /// overridden by `XCOLORRANGE=FULL` / `XCOLORRANGE=LIMITED`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        Self::parse_with_matrix(bytes, ColorMatrix::Bt709)
    }

    /// Parses a Y4M stream with an explicit fallback [`ColorMatrix`] when the
    /// container omits color-matrix tags, separating chroma siting (`C420jpeg` /
    /// `C420mpeg2` / `C420paldv`) from quantization range (`XCOLORRANGE=...`).
    pub fn parse_with_matrix(bytes: &[u8], default_matrix: ColorMatrix) -> Result<Self> {
        let header_end = bytes.iter().position(|&b| b == b'\n').context("Y4M missing header newline")?;
        let header = std::str::from_utf8(&bytes[..header_end]).context("Y4M header not UTF-8")?;
        let mut tokens = header.split_ascii_whitespace();
        ensure!(tokens.next() == Some("YUV4MPEG2"), "not a YUV4MPEG2 stream");
        let (mut width, mut height) = (0u32, 0u32);
        let (mut fps_num, mut fps_den) = (30u32, 1u32);
        let mut siting = ChromaSiting::Centered;
        let mut range = ColorRange::Limited;
        for tok in tokens {
            if let Some(rest) = tok.strip_prefix('W') {
                width = rest.parse().context("invalid Y4M width")?;
            } else if let Some(rest) = tok.strip_prefix('H') {
                height = rest.parse().context("invalid Y4M height")?;
            } else if let Some(rest) = tok.strip_prefix('F') {
                let (n, d) = rest.split_once(':').context("invalid Y4M frame rate")?;
                fps_num = n.parse().context("invalid Y4M fps numerator")?;
                fps_den = d.parse().context("invalid Y4M fps denominator")?;
                ensure!(fps_num > 0 && fps_den > 0, "Y4M frame rate must be positive");
            } else if let Some(rest) = tok.strip_prefix('C') {
                match rest {
                    "420" | "420jpeg" => siting = ChromaSiting::Centered,
                    "420mpeg2" => siting = ChromaSiting::Left,
                    "420paldv" => siting = ChromaSiting::TopLeft,
                    other => bail!("unsupported Y4M chroma format C{other} (only 420 variants supported)"),
                }
            } else if let Some(val) = tok.strip_prefix("XCOLORRANGE=") {
                if val.eq_ignore_ascii_case("FULL") {
                    range = ColorRange::Full;
                } else if val.eq_ignore_ascii_case("LIMITED") {
                    range = ColorRange::Limited;
                }
            }
        }
        let color_space = ColorSpace::from_parts(default_matrix, range);
        ensure!(
            width >= 2
                && height >= 2
                && width <= MAX_VIDEO_DIM
                && height <= MAX_VIDEO_DIM
                && width.is_multiple_of(2)
                && height.is_multiple_of(2),
            "invalid Y4M dimensions {width}x{height}"
        );
        let y_len = (width as usize).checked_mul(height as usize).context("Y4M frame size overflow")?;
        let uv_len = ((width / 2) as usize).checked_mul((height / 2) as usize).context("Y4M chroma overflow")?;
        let frame_bytes = y_len.checked_add(uv_len * 2).context("Y4M total frame overflow")?;
        let mut frames = Vec::new();
        let mut at = header_end + 1;
        while at < bytes.len() {
            let nl = bytes[at..].iter().position(|&b| b == b'\n').context("truncated Y4M FRAME header")?;
            let line = std::str::from_utf8(&bytes[at..at + nl]).context("Y4M FRAME header not UTF-8")?;
            let line_trimmed = line.trim_end_matches('\r');
            ensure!(
                line_trimmed == "FRAME" || line_trimmed.starts_with("FRAME "),
                "expected Y4M FRAME marker, got {line:?}"
            );
            at += nl + 1;
            let end = at.checked_add(frame_bytes).context("Y4M frame offset overflow")?;
            ensure!(end <= bytes.len(), "truncated Y4M frame payload");
            let y = bytes[at..at + y_len].to_vec();
            let u = bytes[at + y_len..at + y_len + uv_len].to_vec();
            let v = bytes[at + y_len + uv_len..end].to_vec();
            frames.push(Yuv420Frame::with_siting(width, height, color_space, siting, y, u, v)?);
            at = end;
        }
        ensure!(!frames.is_empty(), "Y4M stream contains no frames");
        Ok(Self { width, height, fps_num, fps_den, color_space, siting, frames })
    }
}

/// Memory-efficient Y4M stream backed by a single owned byte buffer and per-frame offsets
/// (zero per-frame allocations or plane copies).
#[derive(Clone, Debug)]
pub struct IndexedY4mStream {
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub color_space: ColorSpace,
    pub siting: ChromaSiting,
    bytes: Vec<u8>,
    frame_offsets: Vec<usize>,
    y_len: usize,
    uv_len: usize,
}

impl IndexedY4mStream {
    pub fn fps(&self) -> f64 {
        if self.fps_den == 0 || self.fps_num == 0 { 30.0 } else { f64::from(self.fps_num) / f64::from(self.fps_den) }
    }

    pub fn frame_count(&self) -> usize {
        self.frame_offsets.len()
    }

    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Indexes `bytes` in-place after validating payload byte budget and dimensions against `max_dim`.
    pub fn parse_with_limits(
        bytes: Vec<u8>,
        default_matrix: ColorMatrix,
        max_dim: u32,
        max_bytes: usize,
    ) -> Result<Self> {
        ensure!(bytes.len() <= max_bytes, "Y4M payload {} bytes exceeds {max_bytes}-byte limit", bytes.len());
        let header_end = bytes.iter().position(|&b| b == b'\n').context("Y4M missing header newline")?;
        let header = std::str::from_utf8(&bytes[..header_end]).context("Y4M header not UTF-8")?;
        let mut tokens = header.split_ascii_whitespace();
        ensure!(tokens.next() == Some("YUV4MPEG2"), "not a YUV4MPEG2 stream");
        let (mut width, mut height) = (0u32, 0u32);
        let (mut fps_num, mut fps_den) = (30u32, 1u32);
        let mut siting = ChromaSiting::Centered;
        let mut range = ColorRange::Limited;
        for tok in tokens {
            if let Some(rest) = tok.strip_prefix('W') {
                width = rest.parse().context("invalid Y4M width")?;
            } else if let Some(rest) = tok.strip_prefix('H') {
                height = rest.parse().context("invalid Y4M height")?;
            } else if let Some(rest) = tok.strip_prefix('F') {
                let (n, d) = rest.split_once(':').context("invalid Y4M frame rate")?;
                fps_num = n.parse().context("invalid Y4M fps numerator")?;
                fps_den = d.parse().context("invalid Y4M fps denominator")?;
                ensure!(fps_num > 0 && fps_den > 0, "Y4M frame rate must be positive");
            } else if let Some(rest) = tok.strip_prefix('C') {
                match rest {
                    "420" | "420jpeg" => siting = ChromaSiting::Centered,
                    "420mpeg2" => siting = ChromaSiting::Left,
                    "420paldv" => siting = ChromaSiting::TopLeft,
                    other => bail!("unsupported Y4M chroma format C{other} (only 420 variants supported)"),
                }
            } else if let Some(val) = tok.strip_prefix("XCOLORRANGE=") {
                if val.eq_ignore_ascii_case("FULL") {
                    range = ColorRange::Full;
                } else if val.eq_ignore_ascii_case("LIMITED") {
                    range = ColorRange::Limited;
                }
            }
        }
        let dim_limit = max_dim.min(MAX_VIDEO_DIM);
        ensure!(
            width >= 2
                && height >= 2
                && width <= dim_limit
                && height <= dim_limit
                && width.is_multiple_of(2)
                && height.is_multiple_of(2),
            "invalid Y4M dimensions {width}x{height} (limit {dim_limit})"
        );
        let color_space = ColorSpace::from_parts(default_matrix, range);
        let y_len = (width as usize).checked_mul(height as usize).context("Y4M frame size overflow")?;
        let uv_len = ((width / 2) as usize).checked_mul((height / 2) as usize).context("Y4M chroma overflow")?;
        let frame_bytes = y_len.checked_add(uv_len * 2).context("Y4M total frame overflow")?;
        let mut frame_offsets = Vec::new();
        let mut at = header_end + 1;
        while at < bytes.len() {
            let nl = bytes[at..].iter().position(|&b| b == b'\n').context("truncated Y4M FRAME header")?;
            let line = std::str::from_utf8(&bytes[at..at + nl]).context("Y4M FRAME header not UTF-8")?;
            let line_trimmed = line.trim_end_matches('\r');
            ensure!(
                line_trimmed == "FRAME" || line_trimmed.starts_with("FRAME "),
                "expected Y4M FRAME marker, got {line:?}"
            );
            at += nl + 1;
            let end = at.checked_add(frame_bytes).context("Y4M frame offset overflow")?;
            ensure!(end <= bytes.len(), "truncated Y4M frame payload");
            frame_offsets.push(at);
            at = end;
        }
        ensure!(!frame_offsets.is_empty(), "Y4M stream contains no frames");
        Ok(Self { width, height, fps_num, fps_den, color_space, siting, bytes, frame_offsets, y_len, uv_len })
    }

    pub fn frame_planes(&self, index: usize) -> (&[u8], &[u8], &[u8]) {
        let off = self.frame_offsets[index];
        let y = &self.bytes[off..off + self.y_len];
        let u = &self.bytes[off + self.y_len..off + self.y_len + self.uv_len];
        let v = &self.bytes[off + self.y_len + self.uv_len..off + self.y_len + 2 * self.uv_len];
        (y, u, v)
    }
}

/// One compressed elementary frame extracted from an IVF (`DKIF`) container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IvfFrame {
    pub pts: u64,
    pub payload: Vec<u8>,
}

/// Demuxed IVF (`DKIF`) video container (standard wrapper for VP8/VP9/AV1 elementary streams).
#[derive(Clone, Debug)]
pub struct IvfStream {
    pub fourcc: [u8; 4],
    pub width: u16,
    pub height: u16,
    pub timebase_den: u32,
    pub timebase_num: u32,
    pub frames: Vec<IvfFrame>,
}

impl IvfStream {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() >= 32, "IVF stream shorter than 32-byte header");
        ensure!(&bytes[0..4] == b"DKIF", "not an IVF (DKIF) stream");
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        let header_len = usize::from(u16::from_le_bytes([bytes[6], bytes[7]]));
        ensure!(version == 0 && header_len >= 32 && header_len <= bytes.len(), "invalid IVF header");
        let fourcc = [bytes[8], bytes[9], bytes[10], bytes[11]];
        let width = u16::from_le_bytes([bytes[12], bytes[13]]);
        let height = u16::from_le_bytes([bytes[14], bytes[15]]);
        ensure!(
            width > 0 && height > 0 && u32::from(width) <= MAX_VIDEO_DIM && u32::from(height) <= MAX_VIDEO_DIM,
            "invalid IVF dimensions"
        );
        let timebase_den = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let timebase_num = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        ensure!(timebase_den > 0 && timebase_num > 0, "IVF timebase numerator and denominator must be non-zero");
        let mut frames = Vec::new();
        let mut at = header_len;
        while at < bytes.len() {
            ensure!(at + 12 <= bytes.len(), "truncated IVF frame header");
            let size = u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize;
            let pts = u64::from_le_bytes(bytes[at + 4..at + 12].try_into().unwrap());
            at += 12;
            let end = at.checked_add(size).context("IVF frame size overflow")?;
            ensure!(end <= bytes.len(), "truncated IVF frame payload");
            frames.push(IvfFrame { pts, payload: bytes[at..end].to_vec() });
            at = end;
        }
        Ok(Self { fourcc, width, height, timebase_den, timebase_num, frames })
    }
}

enum VideoSource {
    Owned(Y4mStream),
    Indexed(IndexedY4mStream),
}

/// Wall-clock video stream player tracking current frame index, looping, and pause.
pub struct VideoPlayer {
    source: VideoSource,
    elapsed: f64,
    frame_index: usize,
    pub looping: bool,
    pub paused: bool,
}

impl VideoPlayer {
    pub fn from_y4m(stream: Y4mStream, looping: bool) -> Result<Self> {
        ensure!(!stream.frames.is_empty(), "VideoPlayer requires at least one frame");
        Ok(Self { source: VideoSource::Owned(stream), elapsed: 0.0, frame_index: 0, looping, paused: false })
    }

    pub fn from_indexed_y4m(stream: IndexedY4mStream, looping: bool) -> Result<Self> {
        ensure!(stream.frame_count() > 0, "VideoPlayer requires at least one frame");
        Ok(Self { source: VideoSource::Indexed(stream), elapsed: 0.0, frame_index: 0, looping, paused: false })
    }

    pub fn fps(&self) -> f64 {
        match &self.source {
            VideoSource::Owned(s) => s.fps(),
            VideoSource::Indexed(s) => s.fps(),
        }
    }

    pub fn frame_count(&self) -> usize {
        match &self.source {
            VideoSource::Owned(s) => s.frames.len(),
            VideoSource::Indexed(s) => s.frame_count(),
        }
    }

    pub fn byte_len(&self) -> usize {
        match &self.source {
            VideoSource::Owned(s) => s.frames.iter().map(|f| f.y.len() + f.u.len() + f.v.len()).sum(),
            VideoSource::Indexed(s) => s.byte_len(),
        }
    }

    pub fn current_index(&self) -> usize {
        self.frame_index
    }

    pub fn current_frame(&self) -> &Yuv420Frame {
        match &self.source {
            VideoSource::Owned(s) => &s.frames[self.frame_index],
            VideoSource::Indexed(_) => {
                panic!("current_frame() requires Owned Y4mStream; use current_planes() for IndexedY4mStream")
            }
        }
    }

    /// Zero-copy planar view `(width, height, color_space, siting, y, u, v)` of the current frame.
    pub fn current_planes(&self) -> (u32, u32, ColorSpace, ChromaSiting, &[u8], &[u8], &[u8]) {
        match &self.source {
            VideoSource::Owned(s) => {
                let f = &s.frames[self.frame_index];
                (f.width, f.height, f.color_space, f.siting, &f.y, &f.u, &f.v)
            }
            VideoSource::Indexed(s) => {
                let (y, u, v) = s.frame_planes(self.frame_index);
                (s.width, s.height, s.color_space, s.siting, y, u, v)
            }
        }
    }

    pub fn seek_frame(&mut self, index: usize) {
        let total = self.frame_count();
        self.frame_index = if self.looping { index % total } else { index.min(total.saturating_sub(1)) };
        self.elapsed = self.frame_index as f64 / self.fps();
    }

    /// Seeks to an absolute timestamp `time_s` scaled by `speed` (defaulting to `1.0` if non-positive),
    /// applying looping modulo in floating-point duration BEFORE integer frame indexing, and honoring
    /// seeks regardless of `paused`. Returns the selected frame index.
    pub fn seek_time(&mut self, time_s: f64, speed: f64) -> usize {
        let effective_speed = if speed.is_finite() && speed > 0.0 { speed } else { 1.0 };
        let scaled = if time_s.is_finite() { time_s.max(0.0) * effective_speed } else { 0.0 };
        let total = self.frame_count();
        let fps = self.fps();
        let duration = total as f64 / fps;
        let next = if self.looping {
            let wrapped = scaled.rem_euclid(duration);
            let idx = (wrapped * fps).floor() as usize;
            idx.min(total - 1)
        } else if scaled >= duration {
            total - 1
        } else {
            ((scaled * fps).floor() as usize).min(total - 1)
        };
        self.frame_index = next;
        self.elapsed = next as f64 / fps;
        next
    }

    /// Advances playback by `dt` seconds and returns `Some(&Yuv420Frame)` whenever
    /// the displayed frame index changes (for owned `Y4mStream` players).
    pub fn advance(&mut self, dt: f64) -> Option<&Yuv420Frame> {
        if self.paused || !dt.is_finite() || dt <= 0.0 {
            return None;
        }
        let next_elapsed = self.elapsed + dt;
        let prev = self.frame_index;
        let total = self.frame_count();
        let fps = self.fps();
        let duration = total as f64 / fps;
        let next = if self.looping {
            let wrapped = next_elapsed.rem_euclid(duration);
            ((wrapped * fps).floor() as usize).min(total - 1)
        } else if next_elapsed >= duration {
            total - 1
        } else {
            ((next_elapsed * fps).floor() as usize).min(total - 1)
        };
        self.elapsed = next_elapsed;
        if next != prev {
            self.frame_index = next;
            Some(self.current_frame())
        } else {
            None
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct YuvUniform {
    colorspace: u32,
    siting: u32,
    chroma_shift_u: f32,
    chroma_shift_v: f32,
}

/// GPU YUV420p plane textures + fullscreen conversion pass producing an `Rgba8Unorm`
/// texture (plus an `Rgba8UnormSrgb` sampling view so materials read decoded linear light).
pub struct VideoTexture {
    width: u32,
    height: u32,
    y_tex: wgpu::Texture,
    u_tex: wgpu::Texture,
    v_tex: wgpu::Texture,
    output: wgpu::Texture,
    output_view: wgpu::TextureView,
    srgb_view: wgpu::TextureView,
    params: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl VideoTexture {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Result<Self> {
        let limit = device.limits().max_texture_dimension_2d.min(MAX_VIDEO_DIM);
        ensure!(
            width >= 2
                && height >= 2
                && width <= limit
                && height <= limit
                && width.is_multiple_of(2)
                && height.is_multiple_of(2),
            "VideoTexture requires even dimensions within 2..={limit}, got {width}x{height}"
        );
        let plane = |label, w, h| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let y_tex = plane("video_y", width, height);
        let u_tex = plane("video_u", width / 2, height / 2);
        let v_tex = plane("video_v", width / 2, height / 2);
        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("video_rgba"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            // Allow hardware sRGB -> linear decode when sampled by PBR/lit scene materials.
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        });
        let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
        let srgb_view = output.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("video_yuv_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("video_yuv_params"),
            size: std::mem::size_of::<YuvUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("video_yuv_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let y_view = y_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let u_view = u_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let v_view = v_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("video_yuv_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&y_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&u_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&v_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: params.as_entire_binding() },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yuv420_to_rgba"),
            source: wgpu::ShaderSource::Wgsl(crate::wgsl!("yuv420_to_rgba.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("video_yuv_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("video_yuv_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Ok(Self { width, height, y_tex, u_tex, v_tex, output, output_view, srgb_view, params, bind_group, pipeline })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.output
    }

    /// Linear-byte `Rgba8Unorm` view (raw encoded R'G'B' texels).
    pub fn view(&self) -> &wgpu::TextureView {
        &self.output_view
    }

    /// sRGB-decoding `Rgba8UnormSrgb` view for scene materials so shader reads linear light.
    pub fn srgb_view(&self) -> &wgpu::TextureView {
        &self.srgb_view
    }

    /// Validates and uploads a planar YUV420p frame and records the GPU conversion pass into `encoder`.
    pub fn upload_and_convert(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &Yuv420Frame,
    ) -> Result<()> {
        frame.validate(MAX_VIDEO_DIM)?;
        self.upload_planes(
            queue,
            encoder,
            frame.width,
            frame.height,
            frame.color_space,
            frame.siting,
            &frame.y,
            &frame.u,
            &frame.v,
        )
    }

    /// Zero-copy planar YUV420p upload and GPU conversion pass into `encoder`.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_planes(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        color_space: ColorSpace,
        siting: ChromaSiting,
        y: &[u8],
        u: &[u8],
        v: &[u8],
    ) -> Result<()> {
        ensure!(
            width == self.width && height == self.height,
            "frame size {width}x{height} does not match VideoTexture {}x{}",
            self.width,
            self.height
        );
        let y_len = (width as usize).checked_mul(height as usize).context("Y plane size overflow")?;
        let uv_len = ((width / 2) as usize).checked_mul((height / 2) as usize).context("UV plane size overflow")?;
        ensure!(y.len() == y_len && u.len() == uv_len && v.len() == uv_len, "YUV420 plane sizes mismatch");
        let write_plane = |tex: &wgpu::Texture, w: u32, h: u32, data: &[u8]| {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        };
        write_plane(&self.y_tex, self.width, self.height, y);
        write_plane(&self.u_tex, self.width / 2, self.height / 2, u);
        write_plane(&self.v_tex, self.width / 2, self.height / 2, v);
        let cw = (self.width / 2) as f32;
        let ch = (self.height / 2) as f32;
        // Shift sample coordinates so hardware bilinear taps match 4:2:0 chroma siting:
        // Centered: texel center aligns with 2x2 luma center -> shift (0, 0).
        // Left (MPEG-2): chroma sits 0.25 chroma texels left of center -> shift (+0.25/cw, 0).
        // TopLeft (PAL-DV): chroma sits 0.25 chroma texels top-left -> shift (+0.25/cw, +0.25/ch).
        let (shift_u, shift_v) = match siting {
            ChromaSiting::Centered => (0.0, 0.0),
            ChromaSiting::Left => (0.25 / cw, 0.0),
            ChromaSiting::TopLeft => (0.25 / cw, 0.25 / ch),
        };
        let uniform = YuvUniform {
            colorspace: color_space as u32,
            siting: siting as u32,
            chroma_shift_u: shift_u,
            chroma_shift_v: shift_v,
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&uniform));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("video_yuv_convert"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.output_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_quadrant_y4m(cs: ColorSpace) -> Vec<u8> {
        let (w, h) = (64usize, 64usize);
        let (c_tag, range_tag) = match cs {
            ColorSpace::Bt601Full | ColorSpace::Bt709Full => ("C420jpeg", "XCOLORRANGE=FULL"),
            ColorSpace::Bt601Limited | ColorSpace::Bt709Limited => ("C420mpeg2", "XCOLORRANGE=LIMITED"),
        };
        let mut out = format!("YUV4MPEG2 W{w} H{h} F30:1 Ip {c_tag} {range_tag}\n").into_bytes();
        for frame_idx in 0..2 {
            out.extend_from_slice(b"FRAME\n");
            let mut y = vec![0u8; w * h];
            let mut u = vec![128u8; (w / 2) * (h / 2)];
            let mut v = vec![128u8; (w / 2) * (h / 2)];
            let quad_rgb = |qx: usize, qy: usize| -> (u8, u8, u8) {
                match ((qx + frame_idx) % 2, qy) {
                    (0, 0) => (220, 30, 30),
                    (1, 0) => (30, 210, 40),
                    (0, 1) => (40, 70, 230),
                    _ => (235, 235, 235),
                }
            };
            for py in 0..h {
                for px in 0..w {
                    let (r, g, b) = quad_rgb(usize::from(px >= w / 2), usize::from(py >= h / 2));
                    let (yv, _, _) = rgb_to_ycbcr(r, g, b, cs);
                    y[py * w + px] = yv;
                }
            }
            for cy in 0..h / 2 {
                for cx in 0..w / 2 {
                    let (r, g, b) = quad_rgb(usize::from(cx >= w / 4), usize::from(cy >= h / 4));
                    let (_, uv, vv) = rgb_to_ycbcr(r, g, b, cs);
                    u[cy * (w / 2) + cx] = uv;
                    v[cy * (w / 2) + cx] = vv;
                }
            }
            out.extend_from_slice(&y);
            out.extend_from_slice(&u);
            out.extend_from_slice(&v);
        }
        out
    }

    #[test]
    fn y4m_ffmpeg_limited_range_exact_markers_and_ivf_validation() {
        // FFmpeg outputs `C420jpeg XCOLORRANGE=LIMITED` with Y=16 for black!
        let mut ffmpeg_black = b"YUV4MPEG2 W16 H16 F25:1 Ip C420jpeg XCOLORRANGE=LIMITED\nFRAME\n".to_vec();
        ffmpeg_black.extend(std::iter::repeat_n(16u8, 16 * 16));
        ffmpeg_black.extend(std::iter::repeat_n(128u8, 8 * 8 * 2));
        let black = Y4mStream::parse(&ffmpeg_black).unwrap();
        assert_eq!(black.color_space, ColorSpace::Bt709Limited);
        assert_eq!(black.siting, ChromaSiting::Centered);

        // Reject malformed FRAMEjunk token.
        let mut bad_frame = b"YUV4MPEG2 W16 H16 F25:1 Ip C420mpeg2\nFRAMEjunk\n".to_vec();
        bad_frame.extend(vec![16u8; 16 * 16 + 128]);
        assert!(Y4mStream::parse(&bad_frame).is_err());

        // Reject empty VideoPlayer stream.
        let mut empty = black.clone();
        empty.frames.clear();
        assert!(VideoPlayer::from_y4m(empty, true).is_err());

        let raw = make_quadrant_y4m(ColorSpace::Bt709Limited);
        let stream = Y4mStream::parse(&raw).unwrap();
        assert_eq!((stream.width, stream.height, stream.frames.len()), (64, 64, 2));
        let mut player = VideoPlayer::from_y4m(stream, true).unwrap();
        assert_eq!(player.current_index(), 0);
        assert!(player.advance(0.01).is_none());
        assert!(player.advance(0.03).is_some());
        assert_eq!(player.current_index(), 1);
        assert!(player.advance(0.04).is_some());
        assert_eq!(player.current_index(), 0);

        // Synthetic IVF stream + zero-timebase rejection.
        let mut ivf = Vec::new();
        ivf.extend_from_slice(b"DKIF");
        ivf.extend_from_slice(&0u16.to_le_bytes());
        ivf.extend_from_slice(&32u16.to_le_bytes());
        ivf.extend_from_slice(b"VP90");
        ivf.extend_from_slice(&320u16.to_le_bytes());
        ivf.extend_from_slice(&180u16.to_le_bytes());
        ivf.extend_from_slice(&30u32.to_le_bytes());
        ivf.extend_from_slice(&1u32.to_le_bytes());
        ivf.extend_from_slice(&2u32.to_le_bytes());
        ivf.extend_from_slice(&0u32.to_le_bytes());
        for (pts, data) in [(0u64, &[1u8, 2, 3][..]), (1u64, &[4u8, 5, 6, 7][..])] {
            ivf.extend_from_slice(&(data.len() as u32).to_le_bytes());
            ivf.extend_from_slice(&pts.to_le_bytes());
            ivf.extend_from_slice(data);
        }
        let parsed = IvfStream::parse(&ivf).unwrap();
        assert_eq!((&parsed.fourcc, parsed.width, parsed.height, parsed.frames.len()), (b"VP90", 320, 180, 2));
        assert_eq!(parsed.frames[1].payload, vec![4, 5, 6, 7]);
        assert!(IvfStream::parse(&ivf[..ivf.len() - 1]).is_err());
        let mut zero_tb = ivf.clone();
        zero_tb[16..20].copy_from_slice(&0u32.to_le_bytes());
        assert!(IvfStream::parse(&zero_tb).is_err(), "zero IVF timebase must be rejected");
    }

    #[test]
    fn video_player_seek_time_loops_accurately_honors_paused_seeks_and_indexed_limits() {
        // 3 frames at 2 fps (duration = 1.5 s): looping seek to 2.0 s -> 0.5 s -> frame 1.
        let mut raw = b"YUV4MPEG2 W4 H4 F2:1 Ip C420jpeg\n".to_vec();
        for f in 0..3u8 {
            raw.extend_from_slice(b"FRAME\n");
            raw.extend(std::iter::repeat_n(40 + f * 60, 16));
            raw.extend(std::iter::repeat_n(128u8, 8));
        }
        // Byte limit and dimension limit reject oversized streams before scanning frames.
        assert!(IndexedY4mStream::parse_with_limits(raw.clone(), ColorMatrix::Bt709, 64, raw.len() - 1).is_err());
        assert!(IndexedY4mStream::parse_with_limits(raw.clone(), ColorMatrix::Bt709, 2, raw.len()).is_err());

        let indexed = IndexedY4mStream::parse_with_limits(raw.clone(), ColorMatrix::Bt709, 64, raw.len()).unwrap();
        assert_eq!(indexed.frame_count(), 3);
        assert_eq!(indexed.frame_planes(1).0[0], 100);
        assert_eq!(indexed.frame_planes(2).0[0], 160);

        let mut player = VideoPlayer::from_indexed_y4m(indexed, true).unwrap();
        // Looping seek to 2.0s (4 frames mod 3 = frame 1).
        assert_eq!(player.seek_time(2.0, 1.0), 1);
        assert_eq!(player.current_index(), 1);
        assert_eq!(player.current_planes().4[0], 100);
        // Large looping timestamp does not saturate/overflow: 1_000_000.0 % 1.5 == 1.0s -> frame 2.
        assert_eq!(player.seek_time(1_000_000.0, 1.0), 2);
        // Paused player still honors explicit timestamp seeks.
        player.paused = true;
        assert_eq!(player.seek_time(1.0, 1.0), 2);
        assert_eq!(player.current_index(), 2);
        // Non-looping clamps at last frame.
        player.looping = false;
        assert_eq!(player.seek_time(10.0, 1.0), 2);
    }

    #[test]
    fn gpu_yuv420_to_rgba_converts_bt709_bt601_and_decodes_srgb_for_materials() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter required for YUV420p witness");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

        for (cs, matrix) in
            [(ColorSpace::Bt709Limited, ColorMatrix::Bt709), (ColorSpace::Bt601Full, ColorMatrix::Bt601)]
        {
            let stream = Y4mStream::parse_with_matrix(&make_quadrant_y4m(cs), matrix).unwrap();
            let frame = &stream.frames[0];
            let video = VideoTexture::new(&device, frame.width, frame.height).unwrap();

            // Mutated / truncated plane must be rejected cleanly before GPU submission.
            let mut broken = frame.clone();
            broken.y.pop();
            let mut bad_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            assert!(video.upload_and_convert(&queue, &mut bad_enc, &broken).is_err());

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            video.upload_and_convert(&queue, &mut encoder, frame).unwrap();
            // Verify sRGB material view exists with Rgba8UnormSrgb format for hardware linearization.
            assert_eq!(video.srgb_view().texture().format(), wgpu::TextureFormat::Rgba8Unorm);

            let bytes_per_row = (frame.width * 4).next_multiple_of(256);
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("video_readback"),
                size: u64::from(bytes_per_row * frame.height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: video.texture(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(frame.height),
                    },
                },
                wgpu::Extent3d { width: frame.width, height: frame.height, depth_or_array_layers: 1 },
            );
            queue.submit(Some(encoder.finish()));
            let slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range().unwrap();
            let pixel = |x: u32, y: u32| -> [u8; 4] {
                let off = (y * bytes_per_row + x * 4) as usize;
                [data[off], data[off + 1], data[off + 2], data[off + 3]]
            };
            let check = |x, y, exp: [u8; 3], label: &str| {
                let got = pixel(x, y);
                for ch in 0..3 {
                    let diff = got[ch].abs_diff(exp[ch]);
                    assert!(diff <= 3, "{cs:?} {label}: got {got:?} expected {exp:?}");
                }
                assert_eq!(got[3], 255);
            };
            check(16, 16, [220, 30, 30], "red");
            check(48, 16, [30, 210, 40], "green");
            check(16, 48, [40, 70, 230], "blue");
            check(48, 48, [235, 235, 235], "white");
            drop(data);
            readback.unmap();
        }
    }
}
