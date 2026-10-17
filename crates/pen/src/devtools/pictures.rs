//! The dev tools' atlas: the texture of a material of their catalog (the
//! one their `pictures` requests name), in whose slots the host draws
//! pictures of the game's frames for the timeline and previews of catalog
//! assets for the asset browser and the material panel. Slots are 128 by 80
//! pixels, 8 to a row: the first 16 hold pictures, the rest previews. Both
//! are scaled down on the GPU and copied into the atlas; the dev tools draw
//! a slot as a sprite of the atlas material.
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use pen_protocol::{Catalog, Color, Draw, Quality, RenderFrame, Vector, devtools_asset};
use pen_render::RenderTarget;

use crate::services::Answer;

/// A slot's size in pixels, and how many are in a row.
pub const SLOT: (u32, u32) = (128, 80);
pub const COLUMNS: u32 = 8;
/// The slots pictures go in, from the first, and how many there are.
pub const PICTURES: u32 = 16;
pub const SLOTS: u32 = 64;
/// Frames between pictures: as often as the history keeps a moment, to
/// which a picture's frame can go back quickly.
const PICTURE_EVERY: u64 = pen_runtime::timeline::MOMENT_INTERVAL;
/// What previews are drawn at before they are scaled into their slot.
const PREVIEW_SIZE: (u32, u32) = (SLOT.0 * 2, SLOT.1 * 2);

/// A preview asked for: what it shows, and in which slot.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Preview {
    kind: u32,
    index: u32,
    other: u32,
    slot: u32,
    drawn: bool,
}

/// Scales images into slots: a blitter for the atlas's format and its
/// scratch images, made for the format frames are read from.
struct Shrink {
    format: wgpu::TextureFormat,
    blitter: wgpu::util::TextureBlitter,
    /// Images to scale through, by size: a picture is halved until it is
    /// within twice its slot, so every texel of it counts.
    scratch: HashMap<(u32, u32), wgpu::Texture>,
    /// A black slot, to clear one before a picture of another shape.
    blank: wgpu::Texture,
}

pub struct Atlas {
    /// The format frames are drawn in.
    format: wgpu::TextureFormat,
    /// The atlas material, once the dev tools name it.
    material: Option<u32>,
    taking: bool,
    /// The pictures: slot and the frame that shows it, oldest first.
    pictures: Vec<(u32, u64)>,
    previews: Vec<Preview>,
    /// The meshes previewed: their bounds, by catalog index.
    bounds: HashMap<u32, ([f32; 3], [f32; 3])>,
    shrink: Option<Shrink>,
    /// What previews are drawn into.
    target: Option<wgpu::Texture>,
}

/// The size that fits `(width, height)` into a slot, its shape kept.
fn fitted((width, height): (u32, u32)) -> (u32, u32) {
    let (width, height) = (f64::from(width.max(1)), f64::from(height.max(1)));
    let scale = (f64::from(SLOT.0) / width).min(f64::from(SLOT.1) / height);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "within a slot")]
    let side = |length: f64, most: u32| ((length * scale).round() as u32).clamp(1, most);
    (side(width, SLOT.0), side(height, SLOT.1))
}

/// The sizes a `size` image is scaled through to `goal`: halved while more
/// than twice as large, then the goal.
fn steps(mut size: (u32, u32), goal: (u32, u32)) -> Vec<(u32, u32)> {
    let mut sizes = Vec::new();
    while size.0 > goal.0 * 2 && size.1 > goal.1 * 2 {
        size = (size.0 / 2, size.1 / 2);
        sizes.push(size);
    }
    sizes.push(goal);
    sizes
}

/// Slot `slot`'s top left in the atlas.
fn origin(slot: u32) -> (u32, u32) {
    (slot % COLUMNS * SLOT.0, slot / COLUMNS * SLOT.1)
}

/// A texture drawn into whole, for previews.
struct Offscreen(wgpu::Texture);

impl RenderTarget for Offscreen {
    fn size(&self) -> (u32, u32) {
        (self.0.width(), self.0.height())
    }

    fn view(&mut self) -> Option<wgpu::TextureView> {
        Some(self.0.create_view(&Default::default()))
    }

    fn present(&mut self) {}
}

fn subtract(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalized(a: [f32; 3]) -> [f32; 3] {
    let length = dot(a, a).sqrt().max(f32::MIN_POSITIVE);
    a.map(|value| value / length)
}

/// A camera at `eye` looking at `target`, +Y up (column-major).
fn look_at(eye: [f32; 3], target: [f32; 3]) -> [f32; 16] {
    let back = normalized(subtract(eye, target));
    let right = normalized(cross([0.0, 1.0, 0.0], back));
    let up = cross(back, right);
    [
        right[0],
        up[0],
        back[0],
        0.0,
        right[1],
        up[1],
        back[1],
        0.0,
        right[2],
        up[2],
        back[2],
        0.0,
        -dot(right, eye),
        -dot(up, eye),
        -dot(back, eye),
        1.0,
    ]
}

/// A perspective projection into WebGPU's depth of 0 to 1 (as Ink's
/// `Mat4.perspective`), `focal` 1 over the tangent of half its view.
fn perspective(focal: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let mut m = [0.0; 16];
    m[0] = focal / aspect;
    m[5] = focal;
    m[10] = -far / (far - near);
    m[11] = -1.0;
    m[14] = -(near * far) / (far - near);
    m
}

/// A frame showing `mesh` drawn with `material`, the mesh's bounds `bounds`
/// filling the view from above and in front, under a light of its own.
fn preview_frame(mesh: u32, material: u32, bounds: ([f32; 3], [f32; 3])) -> RenderFrame {
    let (least, most) = bounds;
    let center = [0, 1, 2].map(|axis| (least[axis] + most[axis]) / 2.0);
    let radius = (dot(subtract(most, least), subtract(most, least)).sqrt() / 2.0).max(1e-3);
    let toward = normalized([0.9, 0.7, 1.3]);
    let distance = radius * 2.6;
    let eye = [0, 1, 2].map(|axis| center[axis] + toward[axis] * distance);
    let aspect = PREVIEW_SIZE.0 as f32 / PREVIEW_SIZE.1 as f32;
    let vector = |x, y, z| Vector { x, y, z };
    RenderFrame {
        view: look_at(eye, center),
        projection: perspective(2.4, aspect, distance - radius * 1.5, distance + radius * 1.5),
        clear: Color { r: 0.06, g: 0.08, b: 0.11, a: 1.0 },
        ambient: vector(0.35, 0.36, 0.42),
        light_direction: vector(-0.45, -0.8, -0.4),
        light_color: vector(1.0, 0.97, 0.92),
        quality: Quality { msaa: 1, fxaa: false, scale: 1.0, filtering: 1, ..Quality::default() },
        draws: vec![Draw { mesh, material, transform: IDENTITY, parameters: None, extra: None }],
        ..RenderFrame::default()
    }
}

/// A mesh drawn where it is.
const IDENTITY: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

impl Atlas {
    pub fn new(format: wgpu::TextureFormat) -> Atlas {
        Atlas {
            format,
            material: None,
            taking: false,
            pictures: Vec::new(),
            previews: Vec::new(),
            bounds: HashMap::new(),
            shrink: None,
            target: None,
        }
    }

    /// The dev tools name their atlas material, and say whether to take
    /// pictures; answers the pictures, each its slot and frame.
    pub fn pictures(&mut self, material: Option<u32>, taking: bool) -> Answer {
        if material.is_some() {
            self.material = material;
        }
        self.taking = taking && self.material.is_some();
        let values = self.pictures.iter().flat_map(|&(slot, frame)| [slot as f32, frame as f32]).collect();
        Answer { values, ..Answer::ok(self.pictures.len() as f32) }
    }

    /// Whether the frame showing at `position` is one to take a picture of.
    pub fn wants_picture(&self, position: u64) -> bool {
        self.taking && position.is_multiple_of(PICTURE_EVERY)
    }

    /// The scaler for frames of `format`, made on first use.
    fn shrink(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> &mut Shrink {
        let format = if self.format.is_srgb() {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            // A frame of a linear format holds its colors encoded already:
            // they go into the atlas as they are.
            wgpu::TextureFormat::Rgba8Unorm
        };
        self.shrink.get_or_insert_with(|| {
            let blitter =
                wgpu::util::TextureBlitterBuilder::new(device, format).sample_type(wgpu::FilterMode::Linear).build();
            let blank = scratch(device, format, SLOT);
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blank slot"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &blank.create_view(&Default::default()),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            queue.submit([encoder.finish()]);
            Shrink { format, blitter, scratch: HashMap::new(), blank }
        })
    }

    /// Scales `image` into slot `slot` of `atlas`, its shape kept and the
    /// rest of the slot black.
    fn put(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas: &wgpu::Texture,
        image: &wgpu::Texture,
        slot: u32,
    ) {
        let shrink = self.shrink(device, queue);
        let goal = fitted((image.width(), image.height()));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("atlas") });
        let mut source = image.create_view(&Default::default());
        let mut last = None;
        for size in steps((image.width(), image.height()), goal) {
            let format = shrink.format;
            let target = shrink.scratch.entry(size).or_insert_with(|| scratch(device, format, size)).clone();
            let view = target.create_view(&Default::default());
            shrink.blitter.copy(device, &mut encoder, &source, &view);
            source = view;
            last = Some(target);
        }
        let (x, y) = origin(slot);
        let at = |x, y| wgpu::TexelCopyTextureInfo {
            texture: atlas,
            mip_level: 0,
            origin: wgpu::Origin3d { x, y, z: 0 },
            aspect: wgpu::TextureAspect::All,
        };
        encoder.copy_texture_to_texture(shrink.blank.as_image_copy(), at(x, y), shrink.blank.size());
        if let Some(last) = last {
            let (left, top) = ((SLOT.0 - goal.0) / 2, (SLOT.1 - goal.1) / 2);
            encoder.copy_texture_to_texture(last.as_image_copy(), at(x + left, y + top), last.size());
        }
        queue.submit([encoder.finish()]);
    }

    /// Takes a picture of the frame drawn in `frame`, showing at
    /// `position`, into the next picture slot (the oldest's, once all are
    /// taken).
    pub fn picture(
        &mut self,
        renderer: &mut pen_render::Renderer,
        catalog: &Catalog,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        frame: &wgpu::Texture,
        position: u64,
    ) -> Result<()> {
        let Some(material) = self.material else { return Ok(()) };
        let Some(atlas) = renderer.material_texture(catalog, material)? else { return Ok(()) };
        self.pictures.retain(|&(_, frame)| frame != position);
        let slot = match self.pictures.len() {
            taken if taken < PICTURES as usize => {
                (0..PICTURES).find(|slot| self.pictures.iter().all(|(used, _)| used != slot))
            }
            _ => Some(self.pictures.remove(0).0),
        };
        let Some(slot) = slot else { return Ok(()) };
        self.put(device, queue, &atlas, frame, slot);
        self.pictures.push((slot, position));
        Ok(())
    }

    /// The dev tools ask for a preview of catalog asset `index` of `kind`
    /// with `other` in slot `slot`: value 1 once it is drawn there.
    pub fn preview(&mut self, kind: u32, index: u32, other: u32, slot: u32) -> Answer {
        if self.material.is_none() || !(PICTURES..SLOTS).contains(&slot) || kind > devtools_asset::MATERIAL {
            return Answer::failed();
        }
        let asked = Preview { kind, index, other, slot, drawn: false };
        match self.previews.iter_mut().find(|preview| preview.slot == slot) {
            Some(preview) if Preview { drawn: false, ..*preview } == asked => {
                Answer::ok(f32::from(u8::from(preview.drawn)))
            }
            Some(preview) => {
                *preview = asked;
                Answer::ok(0.0)
            }
            None => {
                self.previews.push(asked);
                Answer::ok(0.0)
            }
        }
    }

    /// Material `index` changed: its previews are drawn again.
    pub fn changed(&mut self, material: u32) {
        for preview in &mut self.previews {
            let shown = if preview.kind == devtools_asset::MATERIAL { preview.index } else { preview.other };
            if shown == material {
                preview.drawn = false;
            }
        }
    }

    /// Draws the previews not drawn yet, each once everything it draws with
    /// has loaded (until then, it draws them again each frame to load it);
    /// each frame, once the dev tools name their atlas.
    pub fn draw_previews(
        &mut self,
        renderer: &mut pen_render::Renderer,
        catalog: &Catalog,
        assets: &Path,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
    ) -> Result<()> {
        let Some(material) = self.material else { return Ok(()) };
        // Looking for the atlas loads it, for the first pictures too.
        let Some(atlas) = renderer.material_texture(catalog, material)? else { return Ok(()) };
        if self.previews.iter().all(|preview| preview.drawn) {
            return Ok(());
        }
        let format = self.format;
        let target = self
            .target
            .get_or_insert_with(|| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("preview"),
                    size: wgpu::Extent3d { width: PREVIEW_SIZE.0, height: PREVIEW_SIZE.1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
            })
            .clone();
        for at in 0..self.previews.len() {
            let preview = self.previews[at];
            if preview.drawn {
                continue;
            }
            let (mesh, shown) = if preview.kind == devtools_asset::MESH {
                (preview.index, preview.other)
            } else {
                (preview.other, preview.index)
            };
            if mesh as usize >= catalog.meshes.len() || shown as usize >= catalog.materials.len() {
                self.previews[at].drawn = true;
                continue;
            }
            let bounds = match self.bounds.get(&mesh) {
                Some(bounds) => *bounds,
                None => {
                    let path = assets.join(catalog.mesh(mesh));
                    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
                    let bounds = pen_render::mesh_bounds(&bytes)?;
                    self.bounds.insert(mesh, bounds);
                    bounds
                }
            };
            // Drawn with everything loaded before and after: the final image.
            let loaded = renderer.pending() == 0 && renderer.quiet();
            renderer.submit(&preview_frame(mesh, shown, bounds), catalog, &mut Offscreen(target.clone()))?;
            if loaded && renderer.pending() == 0 && renderer.quiet() {
                self.put(device, queue, &atlas, &target, preview.slot);
                self.previews[at].drawn = true;
            }
        }
        Ok(())
    }
}

/// An image to scale through, in `format`.
fn scratch(device: &wgpu::Device, format: wgpu::TextureFormat, (width, height): (u32, u32)) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("atlas scratch"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_pictures_into_slots_and_halves_toward_them() {
        assert_eq!(fitted((1920, 1280)), (120, 80));
        assert_eq!(fitted((1920, 1080)), (128, 72));
        assert_eq!(fitted((10, 1000)), (1, 80));
        assert_eq!(steps((1920, 1280), (120, 80)), [(960, 640), (480, 320), (240, 160), (120, 80)]);
        assert_eq!(steps((200, 100), (128, 64)), [(128, 64)]);
        assert_eq!((origin(0), origin(9), origin(63)), ((0, 0), (128, 80), (896, 560)));
    }

    #[test]
    fn pictures_come_every_moment_and_previews_wait_to_be_drawn() {
        let mut atlas = Atlas::new(wgpu::TextureFormat::Bgra8UnormSrgb);
        assert!(!atlas.wants_picture(30), "no atlas named yet");
        assert_eq!(atlas.pictures(None, true).value, 0.0);
        assert!(!atlas.wants_picture(30));
        atlas.pictures(Some(5), true);
        assert!(atlas.wants_picture(30) && atlas.wants_picture(60) && !atlas.wants_picture(31));
        assert_eq!(atlas.preview(devtools_asset::MATERIAL, 3, 0, 4), Answer::failed(), "a picture's slot");
        assert_eq!(atlas.preview(devtools_asset::MATERIAL, 3, 0, 20), Answer::ok(0.0));
        atlas.previews[0].drawn = true;
        assert_eq!(atlas.preview(devtools_asset::MATERIAL, 3, 0, 20), Answer::ok(1.0));
        atlas.changed(3);
        assert_eq!(atlas.preview(devtools_asset::MATERIAL, 3, 0, 20), Answer::ok(0.0), "drawn again once changed");
        assert_eq!(atlas.preview(devtools_asset::MESH, 1, 3, 20), Answer::ok(0.0));
        assert_eq!(atlas.previews.len(), 1, "a slot holds one preview");
        let frame = preview_frame(0, 0, ([-1.0, 0.0, -1.0], [1.0, 2.0, 1.0]));
        pen_render::validate_frame(&frame).unwrap();
    }
}
