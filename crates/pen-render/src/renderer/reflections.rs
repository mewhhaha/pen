// Reflections beyond the sky: a frame's reflection probes captured (the
// frame's own draws seen from a probe, a few faces of one probe a frame:
// environment/reflection_probes.rs), and the screen-space reflections' light
// (the depth of this frame and last, last frame's picture, the matrices that
// reproject a hit): environment.wgsl reads both through `sky_reflection`.
use pen_protocol::RenderFrame;

use super::frame::stencil;
use super::shadows::scene_floats;
use super::{Drawing, Floats, Item, ReadyEntry, SCENE_BYTES, SCENE_FLOATS, Scratch, State, uniform_buffer};
use crate::environment::{CaptureTargets, Environment, SCREEN_FLOATS, Step, face_projection, face_view};
use crate::matrix::{invert, multiply};

/// Screen-space reflections at quality 1 (low) and 2 (high): the steps of the
/// linear march, the halvings after it, and how far the ray reaches in world
/// units (it fades out over the last 40% of that).
pub(crate) const STEPS: [f32; 2] = [16.0, 32.0];
pub(crate) const REFINEMENTS: [f32; 2] = [4.0, 6.0];
pub(crate) const REACH: [f32; 2] = [20.0, 40.0];
/// How thick a surface the march takes the depth buffer's to be, as a share of its
/// distance from the eye (more where a step crosses more depth).
pub(crate) const THICKNESS: f32 = 0.04;
/// Frames the noise's pattern turns over.
const NOISE_FRAMES: u64 = 16;

/// A probe's capture ready to draw: what it draws into, the scene bind group
/// of each face and which faces they are.
pub(super) struct Capture {
    targets: CaptureTargets,
    faces: Vec<(u32, wgpu::BindGroup)>,
    position: [f32; 3],
    step: Step,
    view: usize,
}

/// The screen-space reflections the frame draws at: 0 for none, else 1 or 2;
/// they need the perspective camera TAA and the depth read assume.
pub(super) fn screen_level(frame: &RenderFrame) -> u32 {
    if frame.projection[11] < -0.5 { frame.quality.ssr.min(2) } else { 0 }
}

impl State {
    /// Plans what the frame captures of the reflection probes: nothing while
    /// assets still load (a cube drawn without them would hold the scene
    /// without them).
    pub(super) fn plan_reflections(&mut self, frame: &RenderFrame) {
        let allowed = !frame.loading && self.pending() == 0 && !self.waiting;
        self.capture = self.environment.plan_reflections(frame, allowed);
    }

    /// Whether every draw the capture sees is ready to draw.
    fn capture_complete(scratch: &Scratch) -> bool {
        let wanted = |item: &Item| item.inset().is_none() && item.views & scratch.capture_bit != 0;
        let items = scratch.items.iter().filter(|item| wanted(item)).count();
        let ready = scratch.ready.iter().filter(|entry| wanted(&scratch.items[entry.item])).count();
        items == ready
    }

    /// Writes the scene uniform and picks the bind group of each face of the
    /// capture the frame draws, and the lights it is lit by; none when
    /// there is none, or when something it would show is not ready.
    pub(super) fn prepare_capture(&mut self, frame: &RenderFrame, scratch: &Scratch, samples: u32) -> Option<Capture> {
        let step = self.capture.clone()?;
        if !Self::capture_complete(scratch) {
            return None;
        }
        let at = &step.probe.position;
        let position = [at.x, at.y, at.z];
        let far = step.probe.far;
        let device = self.device.clone();
        while self.capture_uniforms.len() < 6 {
            self.capture_uniforms.push(uniform_buffer(&device, "capture scene", SCENE_BYTES));
        }
        let base = scene_floats(frame, &self.cascades.0, self.cascades.1);
        let projection = face_projection(far);
        let mut faces = Vec::new();
        for (index, face) in step.faces.clone().enumerate() {
            let mut data: Floats<SCENE_FLOATS> = Floats::new();
            data.extend(base.values[..SCENE_FLOATS].iter().copied());
            data.values[..16].copy_from_slice(&face_view(face, position));
            data.values[16..32].copy_from_slice(&projection);
            // Where the eye is, and no contact shadows: there is no depth to march.
            data.values[76..80].copy_from_slice(&[position[0], position[1], position[2], 1.0]);
            data.values[83] = 0.0;
            let uniform = self.capture_uniforms[index].clone();
            self.queue.write_buffer(&uniform, 0, data.bytes());
            let group = self.capture_scene(&uniform);
            faces.push((face, group));
        }
        self.capture_clusters.update_one(
            &device,
            &self.queue,
            &frame.lights,
            (position, far),
            &self.local,
            super::shadows::local_atlas(&self.shadows),
        );
        let targets = self.environment.capture_targets(&device, samples);
        let view = scratch.capture_bit.trailing_zeros() as usize;
        Some(Capture { targets, faces, position, step, view })
    }

    /// Writes the frame's screen-space reflections into its light and keeps
    /// last frame's picture for the scene to read (`prepare_contact` follows,
    /// which draws the depth), when `level` is above 0 and TAA has a last frame.
    pub(super) fn prepare_screen(&mut self, level: u32) {
        self.reflected = match (level, self.post.as_ref().and_then(|post| post.history())) {
            (1.., Some((view, previous))) => Some((view, previous, self.frames)),
            _ => None,
        };
    }

    /// The light's screen-space part, after the depth's layer is known.
    pub(super) fn write_screen(&mut self, frame: &RenderFrame, level: u32) {
        if level == 0 {
            return;
        }
        let Some(shadows) = self.shadows.as_ref() else { return };
        let which = level as usize - 1;
        let jittered = self.post.as_ref().and_then(|post| post.jittered()).unwrap_or(frame.projection);
        let clip = multiply(&jittered, &frame.view);
        let previous = self.reflected.as_ref().map(|(_, previous, _)| *previous);
        // Last frame's depth and picture are read only if both exist.
        let ready = previous.is_some() && shadows.contact_frames >= 2;
        let mut floats = [0.0f32; SCREEN_FLOATS];
        floats[..4].copy_from_slice(&[STEPS[which], (self.frames % NOISE_FRAMES) as f32, THICKNESS, REACH[which]]);
        floats[4..8].copy_from_slice(&[
            shadows.contact_layer as f32,
            (1 - shadows.contact_layer) as f32,
            if ready { 1.0 } else { 0.0 },
            REFINEMENTS[which],
        ]);
        floats[8..10].copy_from_slice(&[jittered[10], jittered[14]]);
        floats[12..28].copy_from_slice(&clip);
        floats[28..44].copy_from_slice(&invert(&clip));
        floats[44..60].copy_from_slice(&previous.unwrap_or(clip));
        self.environment.set_screen(&self.queue, &floats);
    }
}

/// Draws the capture's faces: each the opaque draws seen from the probe into
/// the capture's target (stenciled as the camera's are), then finished into
/// the scratch cube; with the last face, the cube is filtered into the
/// probe's slot.
#[allow(clippy::too_many_arguments, reason = "the pieces a frame's passes borrow")]
pub(super) fn draw_capture(
    capture: &Capture,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    environment: &mut Environment,
    drawing: &Drawing<'_>,
    ready: &[ReadyEntry],
    list: &[usize],
    lights: &wgpu::BindGroup,
) {
    let targets = &capture.targets;
    for (index, (face, group)) in capture.faces.iter().enumerate() {
        {
            let (view, resolve, store) = match &targets.multisampled {
                Some(multisampled) => (multisampled, Some(&targets.resolved), wgpu::StoreOp::Discard),
                None => (&targets.resolved, None, wgpu::StoreOp::Store),
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("reflection capture"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: resolve,
                    // Scene alpha is the share left to the sky.
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                        store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: wgpu::StoreOp::Discard }),
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, group, &[]);
            pass.set_bind_group(2, lights, &[]);
            pass.set_stencil_reference(0);
            let mut mark = 0;
            let mut current: Option<&wgpu::RenderPipeline> = None;
            for &entry_index in list {
                let entry = &ready[entry_index];
                let wanted = stencil(&entry.material.material);
                if mark != wanted {
                    mark = wanted;
                    pass.set_stencil_reference(mark);
                }
                if current != Some(&entry.pipeline) {
                    current = Some(&entry.pipeline);
                    pass.set_pipeline(&entry.pipeline);
                }
                pass.set_bind_group(1, &entry.group, &[entry.offset]);
                drawing.draw(&mut pass, entry, capture.view, None);
            }
        }
        environment.finish_face(device, queue, encoder, targets, *face, index as u64, capture.position);
    }
    if capture.step.finishes {
        environment.finish_probe(device, queue, encoder, capture.step.job.slot);
    }
}
