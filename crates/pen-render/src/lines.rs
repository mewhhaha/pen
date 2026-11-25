//! Batched debug lines in world or viewport coordinates. Prepare once per
//! submission, then draw the prepared 3D and 2D layers in their respective passes.

use std::{collections::HashMap, ops::Range};

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use pen_protocol::Matrix;

use crate::{camera_precision::DepthMode, environment_passes::PipelineTarget};

pub const LINES_WGSL: &str = include_str!("shaders/lines.wgsl");

const W_EPSILON: f32 = 1e-6;
const X_RAY_FLAG: f32 = 1.0;
const SCREEN_FLAG: f32 = 2.0;

/// Storage record shared with WGSL. The five 16-byte lanes retain full f32 HDR colors.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LineSegmentGpu {
    pub p0: [f32; 3],
    pub width_px: f32,
    pub p1: [f32; 3],
    pub dash_len: f32,
    pub color0: [f32; 4],
    pub color1: [f32; 4],
    /// Gap, dash offset and feather in pixels, followed by the flags encoded as an integer-valued float.
    pub params: [f32; 4],
}

const _: () = assert!(size_of::<LineSegmentGpu>() == 80);

/// A world-space endpoint for the hardware line list. Colors are linear, straight-alpha RGBA.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LineLayer {
    #[default]
    DepthTested,
    XRay,
    OnTop,
}

#[derive(Clone, Copy, Debug)]
pub struct LineStyle {
    pub width_px: f32,
    pub dash_len_px: f32,
    pub gap_len_px: f32,
    pub dash_offset_px: f32,
    /// Full width of the antialiasing transition around the ideal edge.
    pub feather_px: f32,
    pub layer: LineLayer,
}

impl Default for LineStyle {
    fn default() -> Self {
        Self {
            width_px: 1.0,
            dash_len_px: 0.0,
            gap_len_px: 0.0,
            dash_offset_px: 0.0,
            feather_px: 1.25,
            layer: LineLayer::DepthTested,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct XRayStyle {
    pub alpha: f32,
    pub desaturate: f32,
    /// Optional pixel dash/gap for the occluded portion of otherwise solid lines.
    pub solid_line_dash_px: Option<[f32; 2]>,
}

impl Default for XRayStyle {
    fn default() -> Self {
        Self { alpha: 0.28, desaturate: 0.5, solid_line_dash_px: None }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LineFrame {
    /// Column-major WebGPU view-projection matrix.
    pub view_proj: Matrix,
    /// Size of the active viewport in pixels. Screen-space coordinates start at its top left.
    pub viewport: [f32; 2],
    /// Select LessEqual for Forward-Z or GreaterEqual for Reversed-Z.
    pub world_target: Option<PipelineTarget>,
    /// The color-only pass may use a different color format and sample count.
    pub screen_target: Option<PipelineTarget>,
    pub xray: XRayStyle,
}

#[derive(Clone, Copy)]
struct Segment {
    gpu: LineSegmentGpu,
    continues_path: bool,
}

/// Reuse with [Self::clear] to retain the CPU allocations.
#[derive(Clone, Default)]
pub struct LineBatch {
    world: Vec<Segment>,
    on_top: Vec<Segment>,
    screen: Vec<Segment>,
    fast: Vec<LineVertex>,
}

impl LineBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.world.clear();
        self.on_top.clear();
        self.screen.clear();
        self.fast.clear();
    }

    pub fn has_3d(&self) -> bool {
        !self.world.is_empty() || !self.on_top.is_empty() || !self.fast.is_empty()
    }

    pub fn has_2d(&self) -> bool {
        !self.screen.is_empty()
    }

    pub fn is_empty(&self) -> bool {
        !self.has_3d() && !self.has_2d()
    }

    /// Merges all 3D and 2D segments from `other` into `self`.
    pub fn extend_from(&mut self, other: &LineBatch) {
        self.world.extend_from_slice(&other.world);
        self.on_top.extend_from_slice(&other.on_top);
        self.screen.extend_from_slice(&other.screen);
        self.fast.extend_from_slice(&other.fast);
    }

    /// Returns a clone containing only the 3D layers (`DepthTested`, `XRay`, `OnTop`, and fast `LineList`).
    pub fn only_3d(&self) -> Self {
        Self { world: self.world.clone(), on_top: self.on_top.clone(), screen: Vec::new(), fast: self.fast.clone() }
    }

    /// Returns a clone containing only the 2D screen-space segments.
    pub fn only_2d(&self) -> Self {
        Self { world: Vec::new(), on_top: Vec::new(), screen: self.screen.clone(), fast: Vec::new() }
    }

    /// Append a world-space segment with a linear HDR color gradient.
    pub fn segment(
        &mut self,
        p0: [f32; 3],
        p1: [f32; 3],
        color0: [f32; 4],
        color1: [f32; 4],
        style: LineStyle,
    ) -> &mut Self {
        self.push(p0, p1, [color0, color1], style, false, false);
        self
    }

    /// Dash phase follows projected pixel length through every edge, including the closing edge.
    /// Round caps overlap at joins under straight-alpha blending.
    #[allow(clippy::too_many_arguments, reason = "convenience entry point for common polyline parameters")]
    pub fn polyline(
        &mut self,
        points: &[[f32; 3]],
        width_px: f32,
        color: [f32; 4],
        closed: bool,
        dash_len_px: f32,
        gap_len_px: f32,
        xray: bool,
    ) -> &mut Self {
        let layer = if xray { LineLayer::XRay } else { LineLayer::DepthTested };
        self.polyline_with_style(
            points,
            color,
            closed,
            LineStyle { width_px, dash_len_px, gap_len_px, layer, ..Default::default() },
        )
    }

    pub fn polyline_with_style(
        &mut self,
        points: &[[f32; 3]],
        color: [f32; 4],
        closed: bool,
        style: LineStyle,
    ) -> &mut Self {
        self.path(points.iter().copied(), color, closed, style, false)
    }

    pub fn segment_2d(
        &mut self,
        p0: [f32; 2],
        p1: [f32; 2],
        color0: [f32; 4],
        color1: [f32; 4],
        style: LineStyle,
    ) -> &mut Self {
        self.push(screen_point(p0), screen_point(p1), [color0, color1], style, true, false);
        self
    }

    pub fn polyline_2d(&mut self, points: &[[f32; 2]], color: [f32; 4], closed: bool, style: LineStyle) -> &mut Self {
        self.path(points.iter().copied().map(screen_point), color, closed, style, true)
    }

    pub fn aabb(&mut self, min: [f32; 3], max: [f32; 3], color: [f32; 4], style: LineStyle) -> &mut Self {
        let corners: [[f32; 3]; 8] = std::array::from_fn(|mask| {
            std::array::from_fn(|axis| if mask & (1 << axis) == 0 { min[axis] } else { max[axis] })
        });
        for axis in 0..3 {
            let bit = 1 << axis;
            for mask in 0..8 {
                if mask & bit == 0 {
                    self.segment(corners[mask], corners[mask | bit], color, color, style);
                }
            }
        }
        self
    }

    /// Append a circle in an arbitrary plane. Requires a finite nonzero normal,
    /// positive radius, and at least three sides. Invalid circles append no edges.
    pub fn circle(
        &mut self,
        center: [f32; 3],
        normal: [f32; 3],
        radius: f32,
        sides: u32,
        color: [f32; 4],
        style: LineStyle,
    ) -> &mut Self {
        let length = normal[0].hypot(normal[1]).hypot(normal[2]);
        if sides < 3 || !radius.is_finite() || radius <= 0.0 || !length.is_finite() || length == 0.0 {
            return self;
        }
        let n = normal.map(|x| x / length);
        let helper = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let u = cross(n, helper);
        let u_len = u[0].hypot(u[1]).hypot(u[2]);
        let u = u.map(|x| x / u_len);
        let v = cross(n, u);
        let points = (0..sides).map(|i| {
            let (sin, cos) = (std::f32::consts::TAU * i as f32 / sides as f32).sin_cos();
            std::array::from_fn(|axis| center[axis] + radius * (cos * u[axis] + sin * v[axis]))
        });
        self.path(points, color, true, style, false)
    }

    pub fn axes(&mut self, origin: [f32; 3], scale: f32, style: LineStyle) -> &mut Self {
        for axis in 0..3 {
            let mut endpoint = origin;
            endpoint[axis] += scale;
            let mut color = [0.0, 0.0, 0.0, 1.0];
            color[axis] = 1.0;
            self.segment(origin, endpoint, color, color, style);
        }
        self
    }

    /// Append endpoints for independent, depth-tested 1px hardware lines.
    /// Every call must provide an even number of vertices.
    pub fn line_list(&mut self, vertices: &[LineVertex]) -> Result<&mut Self> {
        anyhow::ensure!(vertices.len().is_multiple_of(2), "line list requires endpoint pairs");
        self.fast.extend_from_slice(vertices);
        Ok(self)
    }

    fn path(
        &mut self,
        points: impl Iterator<Item = [f32; 3]>,
        color: [f32; 4],
        closed: bool,
        style: LineStyle,
        screen: bool,
    ) -> &mut Self {
        let mut points = points;
        let Some(first) = points.next() else { return self };
        let mut last = first;
        let mut count = 1;
        for point in points {
            self.push(last, point, [color; 2], style, screen, count > 1);
            last = point;
            count += 1;
        }
        if closed && count > 2 && last != first {
            self.push(last, first, [color; 2], style, screen, true);
        }
        self
    }

    fn push(
        &mut self,
        p0: [f32; 3],
        p1: [f32; 3],
        colors: [[f32; 4]; 2],
        style: LineStyle,
        screen: bool,
        continues_path: bool,
    ) {
        let (bucket, flags) = if screen {
            (&mut self.screen, SCREEN_FLAG)
        } else {
            match style.layer {
                LineLayer::DepthTested => (&mut self.world, 0.0),
                LineLayer::XRay => (&mut self.world, X_RAY_FLAG),
                LineLayer::OnTop => (&mut self.on_top, 0.0),
            }
        };
        bucket.push(Segment {
            gpu: LineSegmentGpu {
                p0,
                width_px: style.width_px,
                p1,
                dash_len: style.dash_len_px,
                color0: colors[0],
                color1: colors[1],
                params: [style.gap_len_px, style.dash_offset_px, style.feather_px, flags],
            },
            continues_path,
        });
    }
}

fn screen_point([x, y]: [f32; 2]) -> [f32; 3] {
    [x, y, 0.0]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniform {
    view_proj: Matrix,
    viewport: [f32; 2],
    xray_alpha: f32,
    xray_desaturate: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PassKind {
    Occluded,
    World,
    OnTop,
    Screen,
    Fast,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    color: wgpu::TextureFormat,
    depth: Option<wgpu::TextureFormat>,
    samples: u32,
    reversed_z: bool,
    kind: PassKind,
}

impl PipelineKey {
    fn new(target: PipelineTarget, kind: PassKind) -> Self {
        Self {
            color: target.color_format,
            depth: target.depth_format,
            samples: target.sample_count,
            reversed_z: target.depth_compare == wgpu::CompareFunction::GreaterEqual,
            kind,
        }
    }

    fn depth_state(self) -> Option<wgpu::DepthStencilState> {
        let mode = if self.reversed_z { DepthMode::ReversedZ } else { DepthMode::ForwardZ };
        self.depth.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(false),
            depth_compare: Some(match self.kind {
                PassKind::Occluded => mode.compare_farther(),
                PassKind::OnTop => wgpu::CompareFunction::Always,
                _ => mode.compare_nearer_or_equal(),
            }),
            stencil: Default::default(),
            bias: Default::default(),
        })
    }
}

pub struct LineRenderer {
    shader: wgpu::ShaderModule,
    bind_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
    bind_group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    segments: wgpu::Buffer,
    fast: wgpu::Buffer,
    staging: Vec<LineSegmentGpu>,
}

/// The uploaded batch. Render in passes matching the targets supplied in [LineFrame].
/// Submit every command buffer that uses this batch before preparing another batch
/// on the same renderer, since the GPU buffers are reused.
pub struct PreparedLines<'a> {
    renderer: &'a LineRenderer,
    world: [(Option<PipelineKey>, Range<u32>); 4],
    screen: (Option<PipelineKey>, Range<u32>),
}

impl LineRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lines"),
            source: wgpu::ShaderSource::Wgsl(LINES_WGSL.into()),
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lines"),
            entries: &[
                buffer_entry(
                    0,
                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                    wgpu::BufferBindingType::Uniform,
                    size_of::<Uniform>(),
                ),
                buffer_entry(
                    1,
                    wgpu::ShaderStages::VERTEX,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    size_of::<LineSegmentGpu>(),
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lines"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let uniform = upload_buffer(device, "line uniform", size_of::<Uniform>() as u64, wgpu::BufferUsages::UNIFORM);
        let segments =
            upload_buffer(device, "line segments", size_of::<LineSegmentGpu>() as u64, wgpu::BufferUsages::STORAGE);
        let fast = upload_buffer(device, "hardware lines", size_of::<LineVertex>() as u64, wgpu::BufferUsages::VERTEX);
        let bind_group = bind_group(device, &bind_layout, &uniform, &segments);
        Self {
            shader,
            bind_layout,
            pipeline_layout,
            pipelines: HashMap::new(),
            bind_group,
            uniform,
            segments,
            fast,
            staging: Vec::new(),
        }
    }

    /// Upload line data and prepare the pipelines needed by this batch before recording render passes.
    pub fn prepare<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: LineFrame,
        batch: &LineBatch,
    ) -> Result<PreparedLines<'a>> {
        let depth = validate_frame(frame, batch)?;
        let xray_dash = frame.xray.solid_line_dash_px.unwrap_or([0.0; 2]);
        let xray_count = if frame.xray.alpha > 0.0 {
            batch.world.iter().filter(|s| s.gpu.params[3] == X_RAY_FLAG).count()
        } else {
            0
        };
        let count = batch.world.len() + xray_count + batch.on_top.len() + batch.screen.len();
        let segment_bytes = count as u64 * size_of::<LineSegmentGpu>() as u64;
        let fast_bytes = size_of_val(batch.fast.as_slice()) as u64;
        let limits = device.limits();
        let storage_limit = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
        anyhow::ensure!(segment_bytes <= storage_limit, "line segment batch exceeds the device storage buffer limit");
        anyhow::ensure!(fast_bytes <= limits.max_buffer_size, "hardware line list exceeds the device buffer limit");
        let fast_count = u32::try_from(batch.fast.len())?;

        self.staging.clear();
        let world = stage(&mut self.staging, &batch.world, frame, depth, None)?;
        let occluded =
            if xray_count == 0 { 0..0 } else { stage(&mut self.staging, &batch.world, frame, depth, Some(xray_dash))? };
        let on_top = stage(&mut self.staging, &batch.on_top, frame, depth, None)?;
        let screen = stage(&mut self.staging, &batch.screen, frame, depth, None)?;
        if segment_bytes > self.segments.size() {
            self.segments = upload_buffer(
                device,
                "line segments",
                growth(segment_bytes, storage_limit),
                wgpu::BufferUsages::STORAGE,
            );
            self.bind_group = bind_group(device, &self.bind_layout, &self.uniform, &self.segments);
        }
        if fast_bytes > self.fast.size() {
            self.fast = upload_buffer(
                device,
                "hardware lines",
                growth(fast_bytes, limits.max_buffer_size),
                wgpu::BufferUsages::VERTEX,
            );
        }
        if segment_bytes != 0 {
            queue.write_buffer(&self.segments, 0, bytemuck::cast_slice(&self.staging));
        }
        if fast_bytes != 0 {
            queue.write_buffer(&self.fast, 0, bytemuck::cast_slice(&batch.fast));
        }
        if segment_bytes != 0 || fast_bytes != 0 {
            let uniform = Uniform {
                view_proj: frame.view_proj,
                viewport: frame.viewport,
                xray_alpha: frame.xray.alpha,
                xray_desaturate: frame.xray.desaturate,
            };
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        }

        let world = [
            (self.pipeline(device, frame.world_target, PassKind::Occluded, &occluded), occluded),
            (self.pipeline(device, frame.world_target, PassKind::Fast, &(0..fast_count)), 0..fast_count),
            (self.pipeline(device, frame.world_target, PassKind::World, &world), world),
            (self.pipeline(device, frame.world_target, PassKind::OnTop, &on_top), on_top),
        ];
        let screen = (self.pipeline(device, frame.screen_target, PassKind::Screen, &screen), screen);
        Ok(PreparedLines { renderer: self, world, screen })
    }

    fn pipeline(
        &mut self,
        device: &wgpu::Device,
        target: Option<PipelineTarget>,
        kind: PassKind,
        range: &Range<u32>,
    ) -> Option<PipelineKey> {
        if range.is_empty() {
            return None;
        }
        let key = PipelineKey::new(target.expect("validated line target"), kind);
        self.pipelines.entry(key).or_insert_with(|| create_pipeline(device, &self.shader, &self.pipeline_layout, key));
        Some(key)
    }
}

impl PreparedLines<'_> {
    /// Draw occluded X-ray, hardware lines, depth-tested lines and the on-top layer in that order.
    pub fn render_3d(&self, pass: &mut wgpu::RenderPass<'_>) {
        for draw in &self.world {
            self.render(pass, draw);
        }
    }

    /// Draw pixel-coordinate lines into a color-only render pass.
    pub fn render_2d(&self, pass: &mut wgpu::RenderPass<'_>) {
        self.render(pass, &self.screen);
    }

    fn render(&self, pass: &mut wgpu::RenderPass<'_>, (key, range): &(Option<PipelineKey>, Range<u32>)) {
        let Some(key) = key else { return };
        pass.set_pipeline(&self.renderer.pipelines[key]);
        pass.set_bind_group(0, &self.renderer.bind_group, &[]);
        if key.kind == PassKind::Fast {
            pass.set_vertex_buffer(0, self.renderer.fast.slice(..));
            pass.draw(range.clone(), 0..1);
        } else {
            pass.draw(0..8, range.clone());
        }
    }
}

fn validate_frame(frame: LineFrame, batch: &LineBatch) -> Result<DepthMode> {
    use wgpu::CompareFunction::{GreaterEqual, LessEqual};

    anyhow::ensure!(
        frame.viewport.iter().all(|v| v.is_finite() && *v > 0.0),
        "line viewport must have finite positive dimensions"
    );
    anyhow::ensure!(frame.view_proj.iter().all(|v| v.is_finite()), "line view-projection matrix must be finite");
    anyhow::ensure!((0.0..=1.0).contains(&frame.xray.alpha), "X-ray alpha must be in [0, 1]");
    anyhow::ensure!((0.0..=1.0).contains(&frame.xray.desaturate), "X-ray desaturation must be in [0, 1]");
    if let Some([dash, gap]) = frame.xray.solid_line_dash_px {
        anyhow::ensure!(
            dash > 0.0 && gap > 0.0 && (dash + gap).is_finite(),
            "X-ray dash and gap must be finite and positive"
        );
    }
    let mut depth = DepthMode::ForwardZ;
    if let Some(target) = frame.world_target {
        anyhow::ensure!(
            target.depth_format.is_some_and(|format| format.has_depth_aspect()),
            "3D lines require a depth attachment"
        );
        depth = match target.depth_compare {
            LessEqual => DepthMode::ForwardZ,
            GreaterEqual => DepthMode::ReversedZ,
            _ => anyhow::bail!("3D lines require LessEqual or GreaterEqual as the target depth comparison"),
        };
    } else {
        anyhow::ensure!(
            batch.world.is_empty() && batch.on_top.is_empty() && batch.fast.is_empty(),
            "missing 3D line target"
        );
    }
    if let Some(target) = frame.screen_target {
        anyhow::ensure!(target.depth_format.is_none(), "2D lines require a color-only target");
    } else {
        anyhow::ensure!(batch.screen.is_empty(), "missing 2D line target");
    }
    for target in [frame.world_target, frame.screen_target].into_iter().flatten() {
        anyhow::ensure!(target.sample_count.is_power_of_two(), "invalid line target sample count");
    }
    anyhow::ensure!(
        batch.fast.iter().all(|v| v.pos.iter().chain(&v.color).all(|lane| lane.is_finite())),
        "hardware line endpoints and colors must be finite"
    );
    Ok(depth)
}

fn stage(
    staging: &mut Vec<LineSegmentGpu>,
    bucket: &[Segment],
    frame: LineFrame,
    depth: DepthMode,
    xray_dash: Option<[f32; 2]>,
) -> Result<Range<u32>> {
    let start = u32::try_from(staging.len())?;
    let mut phase = 0.0;
    for segment in bucket {
        let mut gpu = segment.gpu;
        if !segment.continues_path {
            phase = 0.0;
        }
        if let Some([dash, gap]) = xray_dash {
            if gpu.params[3] != X_RAY_FLAG {
                continue;
            }
            if gpu.dash_len <= 0.0 || gpu.params[0] <= 0.0 {
                gpu.dash_len = dash;
                gpu.params[0] = gap;
            }
        }
        anyhow::ensure!(
            gpu.p0
                .iter()
                .chain(&gpu.p1)
                .chain(&gpu.color0)
                .chain(&gpu.color1)
                .chain(&gpu.params)
                .all(|v| v.is_finite())
                && gpu.width_px.is_finite()
                && gpu.width_px > 0.0
                && gpu.dash_len.is_finite()
                && gpu.params[2] >= 0.0,
            "line endpoints, colors and style must be finite with positive width and nonnegative feather"
        );
        let gap = gpu.params[0];
        if gpu.dash_len > 0.0 && gap > 0.0 {
            let period = gpu.dash_len + gap;
            anyhow::ensure!(period.is_finite(), "line dash period must be finite");
            gpu.params[1] = add_phase(gpu.params[1].rem_euclid(period), phase, period);
            let length = if gpu.params[3] == SCREEN_FLAG {
                (gpu.p1[0] - gpu.p0[0]).hypot(gpu.p1[1] - gpu.p0[1])
            } else {
                projected_length(gpu.p0, gpu.p1, frame, depth)
            };
            anyhow::ensure!(length.is_finite(), "projected line length must be finite");
            phase = add_phase(phase, length.rem_euclid(period), period);
        }
        staging.push(gpu);
    }
    Ok(start..u32::try_from(staging.len())?)
}

fn add_phase(a: f32, b: f32, period: f32) -> f32 {
    let wrap_at = period - b;
    if a >= wrap_at {
        return a - wrap_at;
    }
    a + b
}

/// Mirror the shader's homogeneous w, near and far clipping for dash continuity.
/// The projected, depth-clipped arc contributes to phase even outside the viewport.
fn projected_length(p0: [f32; 3], p1: [f32; 3], frame: LineFrame, depth: DepthMode) -> f32 {
    let matrix = frame.view_proj;
    let transform = |p: [f32; 3]| -> [f32; 4] {
        std::array::from_fn(|i| matrix[i] * p[0] + matrix[4 + i] * p[1] + matrix[8 + i] * p[2] + matrix[12 + i])
    };
    let a = transform(p0);
    let b = transform(p1);
    let distances = |p: [f32; 4]| {
        let [near, far] = match depth {
            DepthMode::ForwardZ => [p[2], p[3] - p[2]],
            DepthMode::ReversedZ => [p[3] - p[2], p[2]],
        };
        [p[3] - W_EPSILON, near, far]
    };
    let mut lo = 0.0f32;
    let mut hi = 1.0f32;
    for (d0, d1) in distances(a).into_iter().zip(distances(b)) {
        if d0 < 0.0 && d1 < 0.0 {
            return 0.0;
        }
        if d0 < 0.0 {
            lo = lo.max(d0 / (d0 - d1));
        } else if d1 < 0.0 {
            hi = hi.min(d0 / (d0 - d1));
        }
    }
    if lo > hi {
        return 0.0;
    }
    let lerp = |i: usize, t: f32| a[i] * (1.0 - t) + b[i] * t;
    let w0 = lerp(3, lo).max(W_EPSILON);
    let w1 = lerp(3, hi).max(W_EPSILON);
    let dx = (lerp(0, hi) / w1 - lerp(0, lo) / w0) * (0.5 * frame.viewport[0]);
    let dy = (lerp(1, hi) / w1 - lerp(1, lo) / w0) * (0.5 * frame.viewport[1]);
    dx.hypot(dy)
}

fn growth(needed: u64, limit: u64) -> u64 {
    needed.checked_next_power_of_two().unwrap_or(limit).min(limit) & !3
}

fn buffer_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BufferBindingType,
    size: usize,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size as u64),
        },
        count: None,
    }
}

fn upload_buffer(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    segments: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("line buffers"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: segments.as_entire_binding() },
        ],
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    key: PipelineKey,
) -> wgpu::RenderPipeline {
    const FAST_ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];
    let fast_layout = [Some(wgpu::VertexBufferLayout {
        array_stride: size_of::<LineVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &FAST_ATTRIBUTES,
    })];
    let (vertex, fragment, topology, buffers) = match key.kind {
        PassKind::Fast => ("vs_fast", "fs_fast", wgpu::PrimitiveTopology::LineList, fast_layout.as_slice()),
        PassKind::Occluded => ("vs_line", "fs_xray", wgpu::PrimitiveTopology::TriangleStrip, &[][..]),
        _ => ("vs_line", "fs_line", wgpu::PrimitiveTopology::TriangleStrip, &[][..]),
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(match key.kind {
            PassKind::Occluded => "lines occluded",
            PassKind::World => "lines depth tested",
            PassKind::OnTop => "lines on top",
            PassKind::Screen => "lines 2D",
            PassKind::Fast => "hardware lines",
        }),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vertex),
            compilation_options: Default::default(),
            buffers,
        },
        primitive: wgpu::PrimitiveState { topology, ..Default::default() },
        depth_stencil: key.depth_state(),
        multisample: wgpu::MultisampleState { count: key.samples, ..Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: key.color,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::{
        camera_precision::{perspective_forward_z, perspective_reversed_z},
        testing,
    };

    const SIDE: u32 = 96;
    const VIEWPORT: [f32; 2] = [96.0; 2];
    const ROW_BYTES: u32 = SIDE * 8;
    const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    const WHITE: [f32; 4] = [1.0; 4];
    const EXTENT: wgpu::Extent3d = wgpu::Extent3d { width: SIDE, height: SIDE, depth_or_array_layers: 1 };

    fn frame(mode: DepthMode) -> LineFrame {
        let fov = std::f32::consts::FRAC_PI_2;
        let (view_proj, world_target) = match mode {
            DepthMode::ForwardZ => {
                (perspective_forward_z(fov, 1.0, 0.1, 100.0), PipelineTarget::standard(COLOR, Some(DEPTH)))
            }
            DepthMode::ReversedZ => {
                (perspective_reversed_z(fov, 1.0, 0.1, Some(100.0)), PipelineTarget::reversed_z(COLOR, Some(DEPTH), 1))
            }
        };
        LineFrame {
            view_proj,
            viewport: VIEWPORT,
            world_target: Some(world_target),
            screen_target: None,
            xray: XRayStyle::default(),
        }
    }

    fn point(x: f32, y: f32, z: f32) -> [f32; 3] {
        [-z * (x / 48.0 - 1.0), -z * (1.0 - y / 48.0), z]
    }

    fn texture(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        sample_count: u32,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("line witness"),
            size: EXTENT,
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | usage,
            view_formats: &[],
        })
    }

    fn draw(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lines: &PreparedLines<'_>,
        target: PipelineTarget,
        occluder: Option<&wgpu::RenderPipeline>,
    ) -> Vec<[f32; 4]> {
        use wgpu::TextureUsages;

        let resolved = texture(device, COLOR, 1, TextureUsages::COPY_SRC);
        let resolved_view = resolved.create_view(&Default::default());
        let msaa_view = (target.sample_count > 1).then(|| {
            texture(device, COLOR, target.sample_count, TextureUsages::empty()).create_view(&Default::default())
        });
        let depth_view = target.depth_format.map(|format| {
            texture(device, format, target.sample_count, TextureUsages::empty()).create_view(&Default::default())
        });
        let reversed = target.depth_compare == wgpu::CompareFunction::GreaterEqual;
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: msaa_view.as_ref().unwrap_or(&resolved_view),
                    resolve_target: msaa_view.as_ref().map(|_| &resolved_view),
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: depth_view.as_ref().map(|view| wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if reversed { 0.0 } else { 1.0 }),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            if let Some(occluder) = occluder {
                pass.set_pipeline(occluder);
                pass.draw(0..4, 0..1);
            }
            if depth_view.is_some() {
                lines.render_3d(&mut pass);
            } else {
                lines.render_2d(&mut pass);
            }
        }

        let size = u64::from(ROW_BYTES * SIDE);
        let copy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("line pixel readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            resolved.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &copy,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW_BYTES),
                    rows_per_image: Some(SIDE),
                },
            },
            EXTENT,
        );
        queue.submit([encoder.finish()]);
        testing::read(device, queue, &copy, size)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|pixel| std::array::from_fn(|i| testing::unhalf(u16::from_le_bytes([pixel[i * 2], pixel[i * 2 + 1]]))))
            .collect()
    }

    fn pixel(image: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
        image[(y * SIDE + x) as usize]
    }

    fn occluder(device: &wgpu::Device) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line witness occluder"),
            source: wgpu::ShaderSource::Wgsl(
                r"
                @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
                    return vec4f(f32(i & 1u), f32(i & 2u) - 1.0, 0.5, 1.0);
                }
                @fragment fn fs() -> @location(0) vec4f {
                    return vec4f(0.0, 0.0, 0.0, 1.0);
                }
            "
                .into(),
            ),
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line witness occluder"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            cache: None,
            multiview_mask: None,
        })
    }

    #[test]
    fn constant_pixel_thickness_and_aa_in_both_depth_modes() {
        let Some((device, queue)) = testing::device() else {
            eprintln!("no GPU adapter; skipping line witness");
            return;
        };
        let mut renderer = LineRenderer::new(&device);
        let mut batch = LineBatch::new();
        let style = LineStyle { width_px: 6.0, ..Default::default() };
        batch.segment(point(14.0, 24.25, -2.0), point(82.0, 24.25, -2.0), WHITE, WHITE, style).segment(
            point(14.0, 66.25, -20.0),
            point(82.0, 66.25, -20.0),
            WHITE,
            WHITE,
            style,
        );

        let mut coverage = Vec::new();
        for mode in [DepthMode::ForwardZ, DepthMode::ReversedZ] {
            let frame = frame(mode);
            let prepared = renderer.prepare(&device, &queue, frame, &batch).unwrap();
            let image = draw(&device, &queue, &prepared, frame.world_target.unwrap(), None);
            for (z, start) in [(-2.0, 18), (-20.0, 60)] {
                let profile: Vec<_> = (start..start + 13).map(|y| pixel(&image, 48, y)[0]).collect();
                let half_max = profile.iter().filter(|&&v| v >= 0.5).count();
                let integrated: f32 = profile.iter().sum();
                let edges: Vec<_> = profile.iter().copied().filter(|v| *v > 0.0 && *v < 1.0).collect();
                eprintln!("{mode:?} at {z}m: {half_max}px at half max, integrated {integrated:.6}px, AA {edges:?}");
                assert!((integrated - 6.0).abs() <= 1.5);
                assert!((5..=7).contains(&half_max));
                assert!(edges.iter().any(|&v| v < 0.5) && edges.iter().any(|&v| v > 0.5));
            }
            coverage.push(image);
        }
        assert_eq!(coverage[0], coverage[1], "forward and reversed depth must cover the same pixels");
    }

    #[test]
    fn xray_only_in_occlusion_and_overlay_above_depth() {
        let Some((device, queue)) = testing::device() else {
            eprintln!("no GPU adapter; skipping line witness");
            return;
        };
        let mut renderer = LineRenderer::new(&device);
        let mut batch = LineBatch::new();
        for (y, layer) in [(24.5, LineLayer::XRay), (42.5, LineLayer::DepthTested), (60.5, LineLayer::OnTop)] {
            batch.segment(
                point(12.0, y, -5.0),
                point(84.0, y, -5.0),
                WHITE,
                WHITE,
                LineStyle { width_px: 4.0, layer, ..Default::default() },
            );
        }
        batch
            .line_list(&[
                LineVertex { pos: point(12.5, 76.5, -5.0), color: WHITE },
                LineVertex { pos: point(84.5, 76.5, -5.0), color: WHITE },
            ])
            .unwrap();
        let occluder = occluder(&device);
        for mode in [DepthMode::ForwardZ, DepthMode::ReversedZ] {
            let frame = frame(mode);
            let prepared = renderer.prepare(&device, &queue, frame, &batch).unwrap();
            let image = draw(&device, &queue, &prepared, frame.world_target.unwrap(), Some(&occluder));
            let pair = |y| [pixel(&image, 24, y)[0], pixel(&image, 72, y)[0]];
            let xray = pair(24);
            let regular = pair(42);
            let on_top = pair(60);
            let fast: Vec<_> = (75..=77).map(pair).collect();
            eprintln!("{mode:?} left/right: X-ray {xray:?}, depth-tested {regular:?}, on-top {on_top:?}");
            eprintln!("{mode:?} hardware rows: {fast:?}");
            assert_eq!(xray[0], 1.0);
            assert!((xray[1] - frame.xray.alpha).abs() < 0.002);
            assert_eq!(regular, [1.0, 0.0]);
            assert_eq!(on_top, [1.0, 1.0]);
            assert_eq!(fast.iter().filter(|row| row[0] > 0.9).count(), 1);
            assert!(fast.iter().all(|row| row[1] == 0.0));

            let mut dashed_frame = frame;
            dashed_frame.xray.solid_line_dash_px = Some([8.0, 6.0]);
            let dashed = renderer.prepare(&device, &queue, dashed_frame, &batch).unwrap();
            let image = draw(&device, &queue, &dashed, frame.world_target.unwrap(), Some(&occluder));
            let dashed_samples = [21, 72, 77].map(|x| pixel(&image, x, 24)[0]);
            assert_eq!(dashed_samples, [1.0, xray[1], 0.0]);
            eprintln!("{mode:?} optional occluded dashes R: {dashed_samples:?}");
        }
    }

    #[test]
    fn dashes_cross_near_plane_and_draw_on_color_only_overlay() {
        let Some((device, queue)) = testing::device() else {
            eprintln!("no GPU adapter; skipping line witness");
            return;
        };
        let mut renderer = LineRenderer::new(&device);
        let mut batch = LineBatch::new();
        batch.polyline(
            &[point(8.0, 18.5, -4.0), point(30.0, 18.5, -4.0), point(82.0, 18.5, -8.0)],
            4.0,
            WHITE,
            false,
            8.0,
            6.0,
            false,
        );
        let solid = LineStyle { width_px: 4.0, ..Default::default() };
        batch.segment([-1.05, point(0.0, 58.5, 2.0)[1], 2.0], point(70.0, 58.5, -5.0), WHITE, WHITE, solid);
        batch.segment(point(12.0, 78.5, 2.0), point(80.0, 78.5, 3.0), WHITE, WHITE, solid);
        let hdr = [2.0, 0.0, 0.0, 1.0];
        batch.polyline_2d(&[[12.0, 30.5], [48.0, 30.5], [48.0, 76.5]], hdr, false, solid).segment_2d(
            [12.5, 12.5],
            [76.5, 12.5],
            WHITE,
            hdr,
            solid,
        );

        let screen_target = PipelineTarget { sample_count: 4, ..PipelineTarget::standard(COLOR, None) };
        let mut frame = frame(DepthMode::ForwardZ);
        frame.screen_target = Some(screen_target);
        let prepared = renderer.prepare(&device, &queue, frame, &batch).unwrap();
        let image = draw(&device, &queue, &prepared, frame.world_target.unwrap(), None);
        assert!(image.iter().flatten().all(|v| v.is_finite()));
        let dash_samples = [12, 19, 25, 34, 38, 48].map(|x| pixel(&image, x, 18)[0]);
        let clipped_span: Vec<_> = (0..SIDE).filter(|&x| pixel(&image, x, 58)[0] > 0.5).collect();
        let overlay = draw(&device, &queue, &prepared, screen_target, None);
        let overlay_samples = [(24, 30), (48, 30), (48, 54), (64, 54)].map(|(x, y)| pixel(&overlay, x, y)[0]);
        let gradient = pixel(&overlay, 44, 12);
        let (first, last, count) = (clipped_span.first(), clipped_span.last(), clipped_span.len());
        eprintln!("Dash R {dash_samples:?}, near-clipped x {first:?}..={last:?} ({count} pixels)");
        eprintln!("4x MSAA overlay R {overlay_samples:?}, gradient {gradient:?}");
        assert_eq!(dash_samples, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        assert_eq!(clipped_span.first().copied(), Some(23));
        assert_eq!(clipped_span.last().copied(), Some(71));
        assert!((0..SIDE).all(|x| pixel(&image, x, 78)[0] == 0.0));
        assert_eq!(overlay_samples, [2.0, 2.0, 2.0, 0.0]);
        assert!((gradient[0] - 1.5).abs() < 0.01 && (gradient[1] - 0.5).abs() < 0.01);
    }
}
