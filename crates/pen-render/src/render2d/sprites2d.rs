//! World-space sprite batches for a single texture atlas.

use bytemuck::{Pod, Zeroable};

use super::{Draw2d, buffer, pipeline, prepared};

pub const SPRITES_2D_WGSL: &str = include_str!("../shaders/render2d_sprites.wgsl");

/// World X points right and world Y points up. Positive rotations are clockwise
/// on screen. Zoom measures screen pixels per world unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera2d {
    pub center: [f32; 2],
    pub zoom: f32,
    pub rotation: f32,
    pub viewport: [f32; 2],
}

impl Camera2d {
    /// Column-major orthographic transform for a positive zoom and viewport.
    pub fn world_to_ndc(self) -> [[f32; 4]; 4] {
        let (s, c) = self.rotation.sin_cos();
        let sx = 2.0 * self.zoom / self.viewport[0];
        let sy = 2.0 * self.zoom / self.viewport[1];
        let [x, y] = self.center;
        let a = c * sx;
        let b = -s * sx;
        let d = s * sy;
        let e = c * sy;
        [[a, d, 0.0, 0.0], [b, e, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [-a * x - b * y, -d * x - e * y, 0.0, 1.0]]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Sprite2d {
    /// World position of the pivot.
    pub pos: [f32; 2],
    pub size: [f32; 2],
    /// Normalized pivot measured from the top left, clockwise radians, painter's layer.
    pub pivot_rot_layer: [f32; 4],
    /// Atlas UV endpoints at the top left and bottom right.
    pub uv_rect: [f32; 4],
    /// Straight linear RGBA tint. RGB greater than one is supported on HDR targets.
    pub color: [f32; 4],
    /// Left, top, right and bottom borders in source texels and destination world units.
    pub nine_slice: [f32; 4],
    /// Horizontal and vertical flip (0 or 1), then full atlas width and height.
    /// Zero dimensions use the bound texture's dimensions.
    pub flags: [f32; 4],
}

impl Default for Sprite2d {
    fn default() -> Self {
        Self {
            pos: [0.0; 2],
            size: [1.0; 2],
            pivot_rot_layer: [0.5, 0.5, 0.0, 0.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            color: [1.0; 4],
            nine_slice: [0.0; 4],
            flags: [0.0; 4],
        }
    }
}

/// Stable painter's ordering. Larger layers draw last. Within a layer,
/// optional Y sorting draws lower world Y last. Ties retain input order.
pub fn sort_sprites(sprites: &mut [Sprite2d], y_sort: bool) {
    // Treat the two representations of zero as the same painter's key.
    let key = |x: f32| x + 0.0;
    sprites.sort_by(|a, b| {
        key(a.pivot_rot_layer[3])
            .total_cmp(&key(b.pivot_rot_layer[3]))
            .then_with(|| if y_sort { key(b.pos[1]).total_cmp(&key(a.pos[1])) } else { std::cmp::Ordering::Equal })
    });
}

pub struct Sprites2dPass {
    pipeline: wgpu::RenderPipeline,
}

impl Sprites2dPass {
    /// Accepts any blendable color target format.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        Self {
            pipeline: pipeline(device, "2D sprites", SPRITES_2D_WGSL, target_format, wgpu::BlendState::ALPHA_BLENDING),
        }
    }

    /// Upload one atlas batch in painter's order. The atlas and sampler must be
    /// filterable bindings. Sampling uses mip level zero to preserve slice borders.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        camera: Camera2d,
        sprites: &[Sprite2d],
        atlas: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> Option<Draw2d> {
        if sprites.is_empty() {
            return None;
        }
        let uniform =
            buffer(device, "2D sprite camera", bytemuck::bytes_of(&camera.world_to_ndc()), wgpu::BufferUsages::UNIFORM);
        let instances =
            buffer(device, "2D sprite instances", bytemuck::cast_slice(sprites), wgpu::BufferUsages::STORAGE);
        Some(prepared(
            device,
            &self.pipeline,
            &[
                uniform.as_entire_binding(),
                instances.as_entire_binding(),
                wgpu::BindingResource::TextureView(atlas),
                wgpu::BindingResource::Sampler(sampler),
            ],
            u32::try_from(sprites.len()).expect("sprite instance count exceeds u32"),
        ))
    }
}
