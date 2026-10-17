// HDR output. Frames are drawn as for an SDR screen, in sRGB-encoded light,
// but on an HDR display the present pass rolls highlights off toward the
// display's peak instead of into white, so they reach past 1. Where the
// surface takes that encoding as it is (extended sRGB: Vulkan, Metal and the
// browser), that is all. Where it wants linear light (scRGB, as Windows does)
// or BT.2100 PQ (HDR10), frames draw into a picture of their own, sprites and
// all, and a last pass decodes it for the surface (`Screen`).
//
// `choose_surface` picks the surface's format and color space from what the
// surface offers, which is all a headless run can check.
use wgpu::{SurfaceColorSpace, SurfaceColorSpaces, TextureFormat};

/// SDR white in nits when the display does not say: BT.2408's reference
/// white for HDR.
pub const REFERENCE_WHITE_NITS: f32 = 203.0;

/// How many times brighter than SDR white an HDR display is taken to reach
/// when it does not say: 1000 nits over the reference white.
pub const DEFAULT_HEADROOM: f32 = 1000.0 / REFERENCE_WHITE_NITS;

/// How frames are shown: the surface's color space, and for HDR how far past
/// SDR white the display reaches and how bright that white is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Display {
    /// `Auto` or `Srgb` for SDR; otherwise `ExtendedSrgb`, `ExtendedSrgbLinear`
    /// (scRGB) or `Bt2100Pq`, the color spaces the renderer writes.
    pub color_space: SurfaceColorSpace,
    /// The display's peak over SDR white (`DisplayHdrInfo::tone_map_headroom`).
    pub headroom: f32,
    /// SDR white in nits, which scRGB and PQ are absolute in.
    pub white_nits: f32,
}

impl Display {
    pub const SDR: Display =
        Display { color_space: SurfaceColorSpace::Auto, headroom: 1.0, white_nits: REFERENCE_WHITE_NITS };

    pub fn is_hdr(&self) -> bool {
        self.color_space.is_hdr()
    }

    /// Whether frames draw into a picture of their own that a last pass
    /// decodes for the surface.
    pub(crate) fn decodes(&self) -> bool {
        matches!(self.color_space, SurfaceColorSpace::ExtendedSrgbLinear | SurfaceColorSpace::Bt2100Pq)
    }

    /// The format frames draw in on a surface of `format`.
    pub(crate) fn drawing_format(&self, format: TextureFormat) -> TextureFormat {
        if self.is_hdr() { TextureFormat::Rgba16Float } else { format }
    }

    /// The display's peak as the present pass writes it: sRGB-encoded,
    /// extended past 1.
    pub(crate) fn encoded_peak(&self) -> f32 {
        let linear = f64::from(self.headroom.max(1.0));
        crate::matrix::narrow(1.055 * linear.powf(1.0 / 2.4) - 0.055)
    }
}

impl Default for Display {
    fn default() -> Self {
        Display::SDR
    }
}

/// The surface format and the display for frames on a surface with
/// `capabilities`, showing on a display described by `info`: with `hdr`
/// asked for and an HDR color space offered, the first of extended sRGB,
/// scRGB and PQ (fewest passes first); otherwise SDR, 8 bits and not sRGB,
/// since the present pass encodes colours itself.
pub fn choose_surface(
    capabilities: &wgpu::SurfaceCapabilities,
    info: &wgpu::DisplayHdrInfo,
    hdr: bool,
) -> (TextureFormat, Display) {
    let headroom = info.tone_map_headroom();
    // A display known to be SDR stays SDR whatever is asked.
    if hdr && headroom.is_none_or(|headroom| headroom > 1.0) {
        let white_nits = info
            .luminance
            .and_then(|luminance| luminance.sdr_white_nits)
            .filter(|nits| nits.is_finite() && *nits > 0.0)
            .unwrap_or(REFERENCE_WHITE_NITS);
        let headroom = headroom.unwrap_or(DEFAULT_HEADROOM);
        for (format, spaces, color_space) in [
            (TextureFormat::Rgba16Float, SurfaceColorSpaces::EXTENDED_SRGB, SurfaceColorSpace::ExtendedSrgb),
            (
                TextureFormat::Rgba16Float,
                SurfaceColorSpaces::EXTENDED_SRGB_LINEAR,
                SurfaceColorSpace::ExtendedSrgbLinear,
            ),
            (TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::BT2100_PQ, SurfaceColorSpace::Bt2100Pq),
        ] {
            if capabilities.color_spaces(format).contains(spaces) {
                return (format, Display { color_space, headroom, white_nits });
            }
        }
    }
    let formats = &capabilities.formats;
    let format = [TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm]
        .into_iter()
        .find(|format| formats.contains(format))
        .or_else(|| formats.iter().copied().find(|format| !format.is_srgb()))
        .or_else(|| formats.first().map(|format| format.remove_srgb_suffix()))
        .unwrap_or(TextureFormat::Bgra8Unorm);
    (format, Display::SDR)
}

/// WGSL: the HDR present pass's `toned`. Up to the shoulder each channel
/// shows as it is, as on an SDR screen; above, it rolls off toward the
/// display's peak rather than toward white.
pub(crate) fn toned_wgsl(peak: f32) -> String {
    format!(
        "fn toned(color: vec3f) -> vec3f {{
  let shoulder = 0.9;
  let peak = {peak:.5};
  let d = peak - shoulder;
  let rolled = peak - d * d / (max(color, vec3f(shoulder)) + d - shoulder);
  return select(color, rolled, color > vec3f(shoulder));
}}
"
    )
}

pub(crate) const DECODE: &str = include_str!("shaders/decode.wgsl");

/// The `Screen` of one surface, which every renderer drawing over its frames
/// shares: behind a mutex natively, where a renderer may move between
/// threads; in a cell in the browser, where it stays on the page's.
#[derive(Clone)]
pub(crate) struct SharedScreen(
    #[cfg(not(target_arch = "wasm32"))] std::sync::Arc<std::sync::Mutex<Screen>>,
    #[cfg(target_arch = "wasm32")] std::rc::Rc<std::cell::RefCell<Screen>>,
);

impl SharedScreen {
    pub fn new(screen: Screen) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        return SharedScreen(std::sync::Arc::new(std::sync::Mutex::new(screen)));
        #[cfg(target_arch = "wasm32")]
        return SharedScreen(std::rc::Rc::new(std::cell::RefCell::new(screen)));
    }

    fn with<T>(&self, work: impl FnOnce(&mut Screen) -> T) -> T {
        #[cfg(not(target_arch = "wasm32"))]
        return work(&mut self.0.lock().unwrap());
        #[cfg(target_arch = "wasm32")]
        return work(&mut self.0.borrow_mut());
    }

    /// See `Screen::view`.
    pub fn view(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: &wgpu::TextureView,
        size: (u32, u32),
    ) -> wgpu::TextureView {
        self.with(|screen| screen.view(device, queue, surface, size))
    }

    /// See `Screen::decode`.
    pub fn decode(&self, encoder: &mut wgpu::CommandEncoder, surface: &wgpu::TextureView) {
        self.with(|screen| screen.decode(encoder, surface));
    }
}

/// Where frames for a decoding display draw, and the pass that decodes that
/// picture onto the surface.
pub(crate) struct Screen {
    display: Display,
    pipeline: Option<wgpu::RenderPipeline>,
    uniform: Option<wgpu::Buffer>,
    /// The picture, its view, and the decode pass's bind group.
    picture: Option<(wgpu::Texture, wgpu::TextureView, wgpu::BindGroup)>,
}

impl Screen {
    /// For a surface of `format` showing `display`.
    pub fn new(device: &wgpu::Device, display: Display, format: TextureFormat) -> Self {
        if !display.decodes() {
            return Screen { display, pipeline: None, uniform: None, picture: None };
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("decode"),
            source: wgpu::ShaderSource::Wgsl(
                DECODE.replace("${fullscreen}", include_str!("shaders/fullscreen.wgsl")).into(),
            ),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("decode"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: None,
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: None,
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("decode"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Screen { display, pipeline: Some(pipeline), uniform: Some(uniform), picture: None }
    }

    /// Where a frame for `surface` (the surface's view, `size` pixels)
    /// draws: the surface itself, or the picture decoded onto it last.
    pub fn view(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: &wgpu::TextureView,
        size: (u32, u32),
    ) -> wgpu::TextureView {
        let (Some(pipeline), Some(uniform)) = (&self.pipeline, &self.uniform) else { return surface.clone() };
        if self.picture.as_ref().is_none_or(|(texture, _, _)| (texture.width(), texture.height()) != size) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("picture"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("decode"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                ],
            });
            let pq = self.display.color_space == SurfaceColorSpace::Bt2100Pq;
            let data = [self.display.white_nits, if pq { 1.0 } else { 0.0 }, 0.0, 0.0f32];
            queue.write_buffer(uniform, 0, bytemuck::cast_slice(&data));
            self.picture = Some((texture, view, group));
        }
        self.picture.as_ref().unwrap().1.clone()
    }

    /// Decodes the picture onto `surface`, when frames draw into one.
    pub fn decode(&self, encoder: &mut wgpu::CommandEncoder, surface: &wgpu::TextureView) {
        let (Some(pipeline), Some((_, _, group))) = (&self.pipeline, &self.picture) else { return };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("decode"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capabilities(formats: &[(TextureFormat, SurfaceColorSpaces)]) -> wgpu::SurfaceCapabilities {
        wgpu::SurfaceCapabilities {
            formats: formats
                .iter()
                .filter(|(_, spaces)| spaces.contains(SurfaceColorSpaces::SRGB))
                .map(|&(format, _)| format)
                .collect(),
            format_capabilities: formats
                .iter()
                .map(|&(format, color_spaces)| wgpu::SurfaceFormatCapabilities { format, color_spaces })
                .collect(),
            ..Default::default()
        }
    }

    fn hdr_display(max_nits: f32, white: f32) -> wgpu::DisplayHdrInfo {
        wgpu::DisplayHdrInfo {
            luminance: Some(wgpu::DisplayLuminance {
                max_nits: Some(max_nits),
                sdr_white_nits: Some(white),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn sdr_unless_hdr_is_asked_for_and_offered() {
        let sdr = SurfaceColorSpaces::SRGB;
        let both = SurfaceColorSpaces::SRGB | SurfaceColorSpaces::EXTENDED_SRGB_LINEAR;
        let surface = capabilities(&[
            (TextureFormat::Bgra8UnormSrgb, sdr),
            (TextureFormat::Bgra8Unorm, sdr),
            (TextureFormat::Rgba16Float, both),
        ]);
        let info = hdr_display(1000.0, 250.0);
        assert_eq!(choose_surface(&surface, &info, false), (TextureFormat::Bgra8Unorm, Display::SDR));
        let (format, display) = choose_surface(&surface, &info, true);
        assert_eq!(format, TextureFormat::Rgba16Float);
        assert_eq!(
            display,
            Display { color_space: SurfaceColorSpace::ExtendedSrgbLinear, headroom: 4.0, white_nits: 250.0 }
        );
        assert!(display.decodes());
        // A display known to be SDR stays so.
        let mut dim = info.clone();
        dim.coarse = Some(wgpu::DisplayCoarseRange { high_dynamic_range: Some(false), ..Default::default() });
        assert_eq!(choose_surface(&surface, &dim, true).1, Display::SDR);
        // Only sRGB formats: the encoding is dropped from the first.
        let srgb_only = capabilities(&[(TextureFormat::Rgba8UnormSrgb, sdr)]);
        assert_eq!(choose_surface(&srgb_only, &info, true).0, TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn hdr_takes_the_fewest_passes_offered() {
        let sdr = SurfaceColorSpaces::SRGB;
        let all = SurfaceColorSpaces::EXTENDED_SRGB | SurfaceColorSpaces::EXTENDED_SRGB_LINEAR;
        // Unknown peak: the default headroom; the reference white.
        let unknown = wgpu::DisplayHdrInfo::default();
        let surface = capabilities(&[(TextureFormat::Bgra8Unorm, sdr), (TextureFormat::Rgba16Float, all)]);
        let (format, display) = choose_surface(&surface, &unknown, true);
        assert_eq!((format, display.color_space), (TextureFormat::Rgba16Float, SurfaceColorSpace::ExtendedSrgb));
        assert_eq!((display.headroom, display.white_nits), (DEFAULT_HEADROOM, REFERENCE_WHITE_NITS));
        assert!(!display.decodes());
        // HDR10 only, on 10-bit.
        let pq = capabilities(&[
            (TextureFormat::Bgra8Unorm, sdr),
            (TextureFormat::Rgb10a2Unorm, sdr | SurfaceColorSpaces::BT2100_PQ),
        ]);
        let (format, display) = choose_surface(&pq, &unknown, true);
        assert_eq!((format, display.color_space), (TextureFormat::Rgb10a2Unorm, SurfaceColorSpace::Bt2100Pq));
        assert_eq!(display.drawing_format(format), TextureFormat::Rgba16Float);
        // Nothing HDR offered.
        let plain = capabilities(&[(TextureFormat::Rgba8Unorm, sdr)]);
        assert_eq!(choose_surface(&plain, &unknown, true), (TextureFormat::Rgba8Unorm, Display::SDR));
    }

    #[test]
    fn the_peak_is_encoded_as_frames_are() {
        assert!((Display::SDR.encoded_peak() - 1.0).abs() < 1e-6);
        let display = Display { color_space: SurfaceColorSpace::ExtendedSrgb, headroom: 4.0, white_nits: 203.0 };
        // sRGB's curve, extended: 4 times white encodes to about 1.82.
        assert!((display.encoded_peak() - 1.8248).abs() < 1e-3, "{}", display.encoded_peak());
        assert!(toned_wgsl(1.5).contains("let peak = 1.50000;"));
    }
}
