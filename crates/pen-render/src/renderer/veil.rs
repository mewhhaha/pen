// The loading veil: the frame's sky, from its horizon up, with a short arc
// of light turning round the middle, over the first frames until what they
// draw has loaded; then it clears.
use pen_protocol::RenderFrame;

use super::{State, Veil, overlay_pipeline, uniform_buffer, vector};
use crate::matrix::narrow;
use crate::shader::PipelineContext;

// The loading veil: the frame's sky, from its horizon up, with a short arc
// of light turning round the middle. It is small, so it compiles at once,
// and lies over the first frames until what they draw has loaded, then
// clears.
const VEIL_SHADER: &str = include_str!("../shaders/veil.wgsl");

// How long (seconds) the veil waits on loads that bring nothing before it
// clears anyway.
const VEIL_PATIENCE: f64 = 12.0;

impl Veil {
    pub(super) fn new(context: &PipelineContext) -> Self {
        let pipeline = overlay_pipeline(context, VEIL_SHADER, "veil");
        let uniform = uniform_buffer(&context.device, "veil", 48);
        let group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("veil"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }],
        });
        Veil { pipeline, uniform, group, rose: None, clearing: None }
    }
}

impl State {
    /// Draws the loading veil over `output` of `size` while it lasts; with
    /// `clear`, over nothing, as the frame drew nothing under it.
    pub(super) fn cover(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        frame: &RenderFrame,
        now: f64,
        (width, height): (u32, u32),
        clear: bool,
    ) {
        let (waiting, arrived) = (self.waiting, self.assets.arrived);
        let Some(veil) = &mut self.veil else { return };
        let rose = *veil.rose.get_or_insert(now);
        if veil.clearing.is_none() && (!waiting || now - rose.max(arrived) > VEIL_PATIENCE) {
            veil.clearing = Some(now);
        }
        let cleared = veil.clearing.map_or(0.0, |clearing| {
            ((now - clearing) / f64::from(self.veil_seconds.unwrap_or(0.0)).max(1e-3)).min(1.0)
        });
        let left = 1.0 - cleared * cleared * (3.0 - 2.0 * cleared);
        if left <= 0.0 {
            self.veil = None;
            return;
        }
        let mut data = Vec::with_capacity(12);
        data.extend(vector(&frame.sky.zenith));
        data.extend(vector(&frame.sky.horizon));
        data.extend([narrow(now - rose), narrow(left), width as f32 / height.max(1) as f32, 0.0]);
        self.queue.write_buffer(&veil.uniform, 0, bytemuck::cast_slice(&data));
        let mut covering = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("veil"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if clear { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load },
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        covering.set_pipeline(&veil.pipeline);
        covering.set_bind_group(0, &veil.group, &[]);
        covering.draw(0..3, 0..1);
    }
}
