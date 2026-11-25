//! A row-major tile grid rendered as one world-space quad.

use bytemuck::{Pod, Zeroable};

use super::{Draw2d, buffer, pipeline, prepared};

pub const TILEMAP_WGSL: &str = include_str!("../shaders/render2d_tilemap.wgsl");

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, Zeroable)]
pub struct TileCell(pub u32);

impl TileCell {
    pub const EMPTY: Self = Self(0);

    /// Atlas IDs are one-based, and zero is transparent. Flip in tile-local
    /// axes, then optionally rotate clockwise by 90 degrees.
    pub fn new(id: u16, flip_x: bool, flip_y: bool, rot90: bool, anim_frames: u8) -> Self {
        assert!(anim_frames < 32, "tile animation has at most 31 frames");
        Self(
            u32::from(id)
                | (u32::from(flip_x) << 16)
                | (u32::from(flip_y) << 17)
                | (u32::from(rot90) << 18)
                | (u32::from(anim_frames) << 19),
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TilemapUniform {
    /// Column-major transform from Camera2d::world_to_ndc.
    pub camera_vp: [[f32; 4]; 4],
    /// Top-left world X/Y, then positive tile width and height in world units.
    pub origin_and_tile_size: [f32; 4],
    /// Grid columns and rows, then atlas columns and rows. Atlas cells have
    /// equal integer texel dimensions, with no gutters.
    pub grid_size_and_atlas: [u32; 4],
    /// Time in seconds, frames per second, two unused lanes, then straight RGBA tint.
    pub anim_time_and_tint: [f32; 8],
}

pub struct TilemapPass {
    pipeline: wgpu::RenderPipeline,
}

impl TilemapPass {
    /// Accepts any blendable color target format.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        Self { pipeline: pipeline(device, "2D tilemap", TILEMAP_WGSL, target_format, wgpu::BlendState::ALPHA_BLENDING) }
    }

    /// Upload the grid once. Update dirty ranges in this buffer with Queue::write_buffer.
    pub fn upload_cells(device: &wgpu::Device, cells: &[TileCell]) -> wgpu::Buffer {
        assert!(!cells.is_empty(), "tile grid must contain cells");
        buffer(
            device,
            "2D tilemap cells",
            bytemuck::cast_slice(cells),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )
    }

    /// The storage binding starts at the first cell of row zero. A 256×256
    /// map uses 256 KiB of cells and generates just six GPU vertices.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        uniform: &TilemapUniform,
        cells: &wgpu::Buffer,
        atlas: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> Draw2d {
        let [width, height, columns, rows] = uniform.grid_size_and_atlas;
        assert!(width > 0 && height > 0 && columns > 0 && rows > 0, "tile grid and atlas must be nonempty");
        assert!(u64::from(width) * u64::from(height) <= cells.size() / 4, "tile buffer is smaller than the grid");
        let uniform = buffer(device, "2D tilemap uniform", bytemuck::bytes_of(uniform), wgpu::BufferUsages::UNIFORM);
        prepared(
            device,
            &self.pipeline,
            &[
                uniform.as_entire_binding(),
                cells.as_entire_binding(),
                wgpu::BindingResource::TextureView(atlas),
                wgpu::BindingResource::Sampler(sampler),
            ],
            1,
        )
    }
}
