// Exposure: how bright the scene's light is shown. A frame asks for a fixed
// exposure in stops, or for auto exposure (eye adaptation): the scene is metered
// on the GPU, and the exposure moves toward what brings its average to middle
// grey, brightening and darkening at the speeds the frame sets.
//
// Metering reads the lit scene, after the sky and fog and TAA and before the
// bloom. The bloom takes the exposure into account (its first pass exposes the
// picture before keeping what is past white, so what spills is what is past white
// on the screen), so metering after it would be circular, and a glow is the
// scene's light spilling, not more of it.
//
// Two compute passes: `meter.wgsl` counts every pixel's luminance into a 256-bin
// histogram in log2 (a bin per sixteenth of a stop over the default sixteen
// stops), and `adapt.wgsl` takes the trimmed average of it and steps the
// exposure, in a four-float buffer that stays on the GPU; the bloom and the
// present pass read the exposure from it, so nothing is read back and nothing
// stalls. With a fixed exposure no pass runs, and the frame writes the buffer.
use bytemuck::{Pod, Zeroable};
use pen_protocol::Exposure as Wanted;

use crate::post::TRANSFER_WGSL;
use crate::renderer::readback::Readbacks;

const BINS: u64 = 256;

/// Stops, at most, either way.
pub(crate) const MAX_STOPS: f32 = 16.0;

const METER: &str = include_str!("shaders/meter.wgsl");
const ADAPT: &str = include_str!("shaders/adapt.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeterUniform {
    range: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdaptUniform {
    range: [f32; 4],
    limits: [f32; 4],
    speeds: [f32; 4],
}

fn meter_uniform(wanted: &Wanted) -> MeterUniform {
    MeterUniform { range: [wanted.range_low, BINS as f32 / (wanted.range_high - wanted.range_low), wanted.centre, 0.0] }
}

fn adapt_uniform(wanted: &Wanted, jump: bool) -> AdaptUniform {
    AdaptUniform {
        range: [wanted.range_low, wanted.range_high, wanted.low, wanted.high],
        limits: [wanted.compensation, wanted.min, wanted.max, wanted.target],
        speeds: [wanted.brighten, wanted.darken, wanted.dt, if jump { 1.0 } else { 0.0 }],
    }
}

struct Pipelines {
    meter: wgpu::ComputePipeline,
    adapt: wgpu::ComputePipeline,
}

/// The metering and the exposure it leaves on the GPU.
pub(crate) struct Exposure {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Exposure in stops, the metered average (log2 luminance), 1 once the buffer
    /// holds an exposure, and the exposure asked for.
    state: wgpu::Buffer,
    histogram: wgpu::Buffer,
    meter: wgpu::Buffer,
    adapt: wgpu::Buffer,
    /// Compiled when the first frame meters.
    pipelines: Option<Pipelines>,
    /// The metering's bind group for the picture it last read.
    picture: Option<(wgpu::TextureView, wgpu::BindGroup)>,
    adapting: Option<wgpu::BindGroup>,
    /// Whether the begun frame meters, and whether an earlier exposure is dropped.
    metering: bool,
    jump: bool,
    /// The exposure of a recent frame, copied out and mapped a few frames later, so
    /// that a host can show it and tests see it; nothing waits for it.
    readbacks: Readbacks<()>,
    reading: Option<usize>,
    latest: Option<f32>,
}

fn buffer(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
}

impl Exposure {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        Self {
            device: device.clone(),
            queue: queue.clone(),
            // No exposure yet: zero stops, which is what a fixed frame shows.
            state: buffer(device, "exposure", 16, storage),
            histogram: buffer(device, "luminance histogram", BINS * 4, storage),
            meter: buffer(device, "metering", size_of::<MeterUniform>() as u64, uniform),
            adapt: buffer(device, "adaptation", size_of::<AdaptUniform>() as u64, uniform),
            pipelines: None,
            picture: None,
            adapting: None,
            metering: false,
            jump: false,
            readbacks: Readbacks::new("exposure read back", 3, device),
            reading: None,
            latest: None,
        }
    }

    /// Takes in the exposures that have come back.
    pub fn take(&mut self) {
        let latest = &mut self.latest;
        self.readbacks.take(|bytes, ()| *latest = Some(bytemuck::cast_slice::<u8, f32>(&bytes[..4])[0]));
    }

    /// The exposure, in stops, of a recent frame.
    pub fn latest(&self) -> Option<f32> {
        self.latest
    }

    /// Copies the frame's exposure out for reading back, if a buffer is free.
    pub fn finish(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.reading = self.readbacks.free(&self.device, 16);
        if let Some(index) = self.reading {
            encoder.copy_buffer_to_buffer(&self.state, 0, self.readbacks.buffer(index), 0, 16);
        }
    }

    /// The frame's commands are submitted: the copy can be mapped.
    pub fn submitted(&mut self) {
        if let Some(index) = self.reading.take() {
            self.readbacks.submitted(index, 16, ());
        }
    }

    /// The exposure in stops (its first float), which the bloom and the present
    /// pass read.
    pub fn state(&self) -> &wgpu::Buffer {
        &self.state
    }

    /// The next metered frame takes the exposure its scene asks for at once, as
    /// the frame after a cut does: for a host that restores or replays a game.
    pub fn reset(&mut self) {
        self.jump = true;
    }

    /// Takes the frame's exposure: a fixed one is written now, a metered one is
    /// set up for `meter`.
    pub fn begin(&mut self, wanted: &Wanted) {
        self.metering = wanted.auto;
        if !wanted.auto {
            // Holding the fixed exposure also holds where auto exposure starts
            // from, should the frame switch it on.
            let stops = wanted.compensation;
            self.queue.write_buffer(&self.state, 0, bytemuck::cast_slice(&[stops, 0.0, 1.0, stops]));
            return;
        }
        let jump = std::mem::take(&mut self.jump) || wanted.cut;
        self.queue.write_buffer(&self.meter, 0, bytemuck::bytes_of(&meter_uniform(wanted)));
        self.queue.write_buffer(&self.adapt, 0, bytemuck::bytes_of(&adapt_uniform(wanted, jump)));
    }

    fn pipelines(&mut self) -> &Pipelines {
        let device = &self.device;
        self.pipelines.get_or_insert_with(|| {
            let make = |label: &str, code: String, entry: &str| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(code.into()),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
            Pipelines {
                meter: make("metering", METER.replace("${transfer}", TRANSFER_WGSL), "bins"),
                adapt: make("adaptation", ADAPT.to_owned(), "exposure_step"),
            }
        })
    }

    /// Encodes the metering of `scene` and the exposure's step, if the begun
    /// frame meters.
    pub fn meter(&mut self, encoder: &mut wgpu::CommandEncoder, scene: &wgpu::TextureView, size: (u32, u32)) {
        if !self.metering {
            return;
        }
        self.pipelines();
        let pipelines = self.pipelines.as_ref().unwrap();
        if self.picture.as_ref().is_none_or(|(kept, _)| kept != scene) {
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("metering"),
                layout: &pipelines.meter.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.meter.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(scene) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.histogram.as_entire_binding() },
                ],
            });
            self.picture = Some((scene.clone(), group));
        }
        let adapting = self.adapting.get_or_insert_with(|| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("adaptation"),
                layout: &pipelines.adapt.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.adapt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.histogram.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.state.as_entire_binding() },
                ],
            })
        });
        encoder.clear_buffer(&self.histogram, 0, None);
        {
            let mut pass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("metering"), timestamp_writes: None });
            pass.set_pipeline(&pipelines.meter);
            pass.set_bind_group(0, &self.picture.as_ref().unwrap().1, &[]);
            pass.dispatch_workgroups(size.0.div_ceil(16), size.1.div_ceil(16), 1);
        }
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("adaptation"), timestamp_writes: None });
        pass.set_pipeline(&pipelines.adapt);
        pass.set_bind_group(0, &*adapting, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

/// The metering in f64, step by step as `meter.wgsl` and `adapt.wgsl` say it: what
/// the shaders are checked against.
#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test pictures, bins and stops"
)]
pub(crate) mod reference {
    use super::Wanted;
    use crate::grade::reference::{luminance, srgb_decode};

    pub const BINS: usize = 256;

    /// A pixel's luminance, from its sRGB-encoded color.
    pub fn luminance_of(pixel: [f64; 3]) -> f64 {
        luminance(pixel.map(srgb_decode))
    }

    pub fn bin_of(light: f64, wanted: &Wanted) -> usize {
        if light > 0.0 {
            let position = (light.log2() - f64::from(wanted.range_low)) * BINS as f64
                / f64::from(wanted.range_high - wanted.range_low);
            position.clamp(0.0, 255.0) as usize
        } else {
            0
        }
    }

    /// The histogram of a `width` by `height` picture of `pixels` (encoded colors).
    pub fn histogram(pixels: &[[f64; 3]], width: usize, height: usize, wanted: &Wanted) -> [u64; BINS] {
        let mut bins = [0u64; BINS];
        for (index, pixel) in pixels.iter().enumerate() {
            let (x, y) = ((index % width) as f64 + 0.5, (index / width) as f64 + 0.5);
            let (dx, dy) = (x / width as f64 - 0.5, y / height as f64 - 0.5);
            let centre = f64::from(wanted.centre);
            let weight = 1.0 + (f64::exp(-6.0 * (dx * dx + dy * dy)) - 1.0) * centre;
            bins[bin_of(luminance_of(*pixel), wanted)] += (weight * 256.0 + 0.5) as u64;
        }
        bins
    }

    /// The log2 luminance average of the part of the weight between the percentiles.
    pub fn average(bins: &[u64; BINS], wanted: &Wanted) -> Option<f64> {
        let total: u64 = bins.iter().sum();
        if total == 0 {
            return None;
        }
        let width = f64::from(wanted.range_high - wanted.range_low) / BINS as f64;
        let (low, high) = (f64::from(wanted.low), f64::from(wanted.high));
        let (mut seen, mut weight, mut logs) = (0u64, 0.0, 0.0);
        for (bin, &count) in bins.iter().enumerate() {
            let before = seen as f64 / total as f64;
            seen += count;
            let after = seen as f64 / total as f64;
            let inside = (after.min(high) - before.max(low)).max(0.0);
            weight += inside;
            logs += inside * (f64::from(wanted.range_low) + (bin as f64 + 0.5) * width);
        }
        Some(logs / weight)
    }

    /// The exposure, in stops, that brings `average` to the target, within the limits.
    pub fn wanted_stops(average: f64, wanted: &Wanted) -> f64 {
        (f64::from(wanted.target).log2() - average + f64::from(wanted.compensation))
            .clamp(f64::from(wanted.min), f64::from(wanted.max))
    }

    /// One step from `current` (none before any) toward `goal`, over `wanted.dt`.
    pub fn step(current: Option<f64>, goal: f64, wanted: &Wanted) -> f64 {
        let stops = match current {
            Some(current) => {
                let rate = f64::from(if goal > current { wanted.brighten } else { wanted.darken });
                current + (goal - current) * (1.0 - (-rate * f64::from(wanted.dt)).exp())
            }
            None => goal,
        };
        stops.clamp(f64::from(wanted.min), f64::from(wanted.max))
    }

    /// The same average without bins: the picture's pixels in order of luminance,
    /// the part between the percentiles, the mean of their log2 luminance.
    pub fn exact_average(pixels: &[[f64; 3]], wanted: &Wanted) -> f64 {
        let mut logs: Vec<f64> =
            pixels.iter().map(|pixel| luminance_of(*pixel).max(f64::from(wanted.range_low).exp2()).log2()).collect();
        logs.sort_by(f64::total_cmp);
        let count = logs.len() as f64;
        let (low, high) = (f64::from(wanted.low) * count, f64::from(wanted.high) * count);
        let (mut weight, mut sum) = (0.0, 0.0);
        for (index, log) in logs.iter().enumerate() {
            let inside = ((index as f64 + 1.0).min(high) - (index as f64).max(low)).max(0.0);
            weight += inside;
            sum += inside * log;
        }
        sum / weight
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test pictures, bins and stops"
)]
mod tests {
    use super::reference::{self, BINS};
    use super::*;
    use crate::grade::reference::srgb_encode;
    use crate::testing::{device, picture, read, unhalf};

    struct Rig {
        device: wgpu::Device,
        queue: wgpu::Queue,
        exposure: Exposure,
    }

    /// What a frame left: the exposure's four floats.
    #[derive(Clone, Copy, Debug)]
    struct State {
        stops: f64,
        average: f64,
        valid: f64,
        wanted: f64,
    }

    fn auto() -> Wanted {
        Wanted { auto: true, dt: 1.0 / 60.0, ..Default::default() }
    }

    /// The encoded grey whose luminance is 2^`stops`.
    fn grey(stops: f64) -> [f32; 4] {
        let value = srgb_encode(stops.exp2()) as f32;
        [value, value, value, 1.0]
    }

    /// The pixels as the half floats the picture holds them in, for the reference.
    fn held(pixels: &[[f32; 4]]) -> Vec<[f64; 3]> {
        pixels.iter().map(|pixel| [0, 1, 2].map(|i| f64::from(unhalf(crate::testing::half(pixel[i]))))).collect()
    }

    /// The middle of bin `bin`, as a luminance in stops (the default range).
    fn centre(bin: usize) -> f64 {
        -10.0 + (bin as f64 + 0.5) / 16.0
    }

    impl Rig {
        fn new() -> Option<Rig> {
            let (device, queue) = device()?;
            let exposure = Exposure::new(&device, &queue);
            Some(Rig { device, queue, exposure })
        }

        fn state(&self) -> State {
            let bytes = read(&self.device, &self.queue, self.exposure.state(), 16);
            let floats: &[f32] = bytemuck::cast_slice(&bytes);
            State {
                stops: f64::from(floats[0]),
                average: f64::from(floats[1]),
                valid: f64::from(floats[2]),
                wanted: f64::from(floats[3]),
            }
        }

        fn histogram(&self) -> Vec<u32> {
            let bytes = read(&self.device, &self.queue, &self.exposure.histogram, BINS as u64 * 4);
            bytemuck::cast_slice(&bytes).to_vec()
        }

        /// One frame of `wanted` over the picture `view` of `size`.
        fn frame(&mut self, wanted: &Wanted, view: &wgpu::TextureView, size: (u32, u32)) -> State {
            self.exposure.begin(wanted);
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.exposure.meter(&mut encoder, view, size);
            self.queue.submit([encoder.finish()]);
            self.state()
        }

        fn view(&self, size: (u32, u32), pixels: &[[f32; 4]]) -> wgpu::TextureView {
            picture(&self.device, &self.queue, size.0, size.1, pixels).create_view(&Default::default())
        }
    }

    /// A picture's two halves (left, right) of these stops of luminance.
    fn halves(size: (u32, u32), left: f64, right: f64) -> Vec<[f32; 4]> {
        (0..size.0 * size.1).map(|i| grey(if i % size.0 < size.0 / 2 { left } else { right })).collect()
    }

    #[test]
    fn the_histogram_counts_each_pixels_log_luminance_as_the_reference_does() {
        let Some(mut rig) = Rig::new() else { return };
        let size = (64, 48);
        let count = (size.0 * size.1) as usize;
        let bright = {
            let mut pixels = vec![grey(centre(120)); count];
            // One pixel of about fifty times SDR white, in the middle.
            pixels[count / 2 + 7] = grey(centre(250));
            pixels
        };
        let cases: [(&str, Vec<[f32; 4]>); 3] = [
            ("uniform", vec![grey(centre(120)); count]),
            ("bimodal", halves(size, centre(40), centre(200))),
            ("a single bright pixel", bright),
        ];
        let wanted = auto();
        for (name, pixels) in cases {
            let view = rig.view(size, &pixels);
            let state = rig.frame(&wanted, &view, size);
            let seen = held(&pixels);
            let bins = reference::histogram(&seen, size.0 as usize, size.1 as usize, &wanted);
            let gpu: Vec<u64> = rig.histogram().into_iter().map(u64::from).collect();
            assert_eq!(gpu, bins, "{name}");
            // The trimmed average and the exposure it asks for.
            let average = reference::average(&bins, &wanted).unwrap();
            assert!((state.average - average).abs() < 1e-4, "{name}: {} vs {average}", state.average);
            let goal = reference::wanted_stops(average, &wanted);
            assert!((state.wanted - goal).abs() < 1e-4, "{name}: {} vs {goal}", state.wanted);
            // A first frame snaps there.
            assert!((state.stops - goal).abs() < 1e-4 && state.valid == 1.0, "{name}: {state:?}");
            // Binning stays within a bin of the exact trimmed mean of the logs.
            let exact = reference::exact_average(&seen, &wanted);
            assert!((average - exact).abs() <= 1.0 / 32.0 + 1e-3, "{name}: binned {average}, exact {exact}");
        }
    }

    #[test]
    fn a_dark_and_a_bright_half_meter_to_their_geometric_mean_until_trimmed() {
        let Some(mut rig) = Rig::new() else { return };
        let size = (64, 32);
        let view = rig.view(size, &halves(size, centre(40), centre(200)));
        let dark = centre(40);
        let light = centre(200);
        let base = auto();
        // Trimming a tenth off each end keeps both halves, equally.
        let both = rig.frame(&base, &view, size);
        assert!((both.average - (dark + light) / 2.0).abs() < 1e-4, "{both:?}");
        // Keeping only above the middle meters the bright half alone, and only below
        // it the dark one.
        let top = rig.frame(&Wanted { low: 0.55, high: 0.95, ..base }, &view, size);
        assert!((top.average - light).abs() < 1e-4, "{top:?}");
        let bottom = rig.frame(&Wanted { low: 0.05, high: 0.45, ..base }, &view, size);
        assert!((bottom.average - dark).abs() < 1e-4, "{bottom:?}");
        // A percentile inside a bin counts the share of it that is inside.
        let share = rig.frame(&Wanted { low: 0.25, high: 0.75, ..base }, &view, size);
        assert!((share.average - (dark + light) / 2.0).abs() < 1e-4, "{share:?}");
        let skew = rig.frame(&Wanted { low: 0.4, high: 0.9, ..base }, &view, size);
        assert!((skew.average - (dark * 0.1 + light * 0.4) / 0.5).abs() < 1e-4, "{skew:?}");
    }

    #[test]
    fn a_lone_bright_pixel_is_trimmed_away_and_centre_weighting_favours_the_middle() {
        let Some(mut rig) = Rig::new() else { return };
        let size = (64, 64);
        let mut pixels = vec![grey(centre(120)); (size.0 * size.1) as usize];
        pixels[(size.0 * 31 + 31) as usize] = grey(centre(250));
        let view = rig.view(size, &pixels);
        let state = rig.frame(&auto(), &view, size);
        assert!((state.average - centre(120)).abs() < 1e-4, "{state:?}");
        // Untrimmed, it pulls the average up by its share of the weight.
        let all = rig.frame(&Wanted { low: 0.0, high: 1.0, ..auto() }, &view, size);
        let pulled = centre(120) + (centre(250) - centre(120)) / f64::from(size.0 * size.1);
        assert!((all.average - pulled).abs() < 1e-4, "{all:?} vs {pulled}");
        // A bright middle among a dark picture: centre weighting meters it brighter.
        let mut pixels = vec![grey(centre(60)); (size.0 * size.1) as usize];
        for y in 24..40 {
            for x in 24..40 {
                pixels[(y * size.0 + x) as usize] = grey(centre(180));
            }
        }
        let view = rig.view(size, &pixels);
        let flat = Wanted { low: 0.0, high: 1.0, ..auto() };
        let even = rig.frame(&flat, &view, size);
        let middle = rig.frame(&Wanted { centre: 1.0, ..flat }, &view, size);
        assert!(middle.average > even.average + 0.2, "{even:?} {middle:?}");
        let seen = held(&pixels);
        let bins = reference::histogram(&seen, 64, 64, &Wanted { centre: 1.0, ..flat });
        let average = reference::average(&bins, &flat).unwrap();
        assert!((middle.average - average).abs() < 5e-3, "{} vs {average}", middle.average);
    }

    #[test]
    fn adaptation_approaches_its_goal_with_the_rate_of_its_direction() {
        let Some(mut rig) = Rig::new() else { return };
        let size = (32, 32);
        let view = rig.view(size, &vec![grey(centre(100)); 1024]);
        let goal = 0.18f64.log2() - centre(100);
        for (name, start, rate) in [("brightening", goal - 3.0, 3.0), ("darkening", goal + 3.0, 1.0)] {
            // The exposure the last frame left, as a fixed frame writes it.
            rig.frame(&Wanted { compensation: start as f32, ..Default::default() }, &view, size);
            let wanted = Wanted { brighten: 3.0, darken: 1.0, dt: 0.05, ..auto() };
            let mut stepped = start;
            for frame in 1..=30 {
                let state = rig.frame(&wanted, &view, size);
                // The closed form of the time constant, and the reference's frame by frame.
                let expected = goal + (start - goal) * (-rate * 0.05 * f64::from(frame)).exp();
                assert!((state.stops - expected).abs() < 2e-4, "{name} frame {frame}: {} vs {expected}", state.stops);
                stepped = reference::step(Some(stepped), goal, &wanted);
                assert!((state.stops - stepped).abs() < 2e-4, "{name} frame {frame}: {} vs {stepped}", state.stops);
            }
        }
    }

    #[test]
    fn splitting_time_into_frames_changes_nothing() {
        let Some(mut rig) = Rig::new() else { return };
        let size = (16, 16);
        let view = rig.view(size, &vec![grey(centre(100)); 256]);
        let goal = 0.18f64.log2() - centre(100);
        let start = goal + 2.5;
        let mut finals = Vec::new();
        for frames in [1u32, 4, 10, 60, 240] {
            rig.frame(&Wanted { compensation: start as f32, ..Default::default() }, &view, size);
            let wanted = Wanted { dt: 1.0 / frames as f32, ..auto() };
            let mut state = rig.state();
            for _ in 0..frames {
                state = rig.frame(&wanted, &view, size);
            }
            finals.push(state.stops);
        }
        let expected = goal + (start - goal) * (-1.0f64).exp();
        for (frames, stops) in [1, 4, 10, 60, 240].into_iter().zip(&finals) {
            assert!((stops - expected).abs() < 3e-4, "{frames} frames: {stops} vs {expected}");
        }
        let (least, most) = finals.iter().fold((f64::MAX, f64::MIN), |(l, m), &v| (l.min(v), m.max(v)));
        assert!(most - least < 3e-4, "{finals:?}");
        // No time passing holds the exposure, brightening or darkening alike.
        let held = rig.frame(&Wanted { dt: 0.0, ..auto() }, &view, size);
        assert!((held.stops - finals[4]).abs() < 1e-6, "{held:?}");
    }

    #[test]
    fn limits_compensation_cuts_and_a_fixed_exposure() {
        let Some(mut rig) = Rig::new() else { return };
        let size = (16, 16);
        let dark = rig.view(size, &vec![grey(-9.5); 256]);
        let bright = rig.view(size, &vec![grey(5.5); 256]);
        // Far darker than middle grey asks for more than the most, and far
        // brighter for less than the least.
        let limits = Wanted { min: -2.0, max: 3.0, ..auto() };
        let up = rig.frame(&limits, &dark, size);
        assert!(up.wanted == 3.0 && up.stops == 3.0, "{up:?}");
        let down = rig.frame(&Wanted { dt: 10.0, ..limits }, &bright, size);
        assert!(down.wanted == -2.0 && (down.stops + 2.0).abs() < 1e-3 && down.stops >= -2.0, "{down:?}");
        // Compensation biases the goal, inside the limits.
        let view = rig.view(size, &vec![grey(centre(100)); 256]);
        let plain = rig.frame(&Wanted { cut: true, ..auto() }, &view, size);
        let biased = rig.frame(&Wanted { cut: true, compensation: 1.5, ..auto() }, &view, size);
        assert!((biased.stops - plain.stops - 1.5).abs() < 1e-4, "{plain:?} {biased:?}");
        let capped = rig.frame(&Wanted { cut: true, compensation: 9.0, ..auto() }, &view, size);
        assert_eq!(capped.stops, 6.0);
        // A cut jumps to the goal at once; without one the exposure only moves toward it.
        let goal = plain.wanted;
        rig.frame(&Wanted { compensation: goal as f32 - 4.0, ..Default::default() }, &view, size);
        let slow = rig.frame(&auto(), &view, size);
        assert!(slow.stops < goal - 3.0, "{slow:?}");
        let jumped = rig.frame(&Wanted { cut: true, ..auto() }, &view, size);
        assert!((jumped.stops - goal).abs() < 1e-4, "{jumped:?}");
        // A reset from the host does the same, once.
        rig.frame(&Wanted { compensation: goal as f32 - 4.0, ..Default::default() }, &view, size);
        rig.exposure.reset();
        let reset = rig.frame(&auto(), &view, size);
        assert!((reset.stops - goal).abs() < 1e-4, "{reset:?}");
        rig.frame(&Wanted { compensation: goal as f32 - 4.0, ..Default::default() }, &view, size);
        let after = rig.frame(&auto(), &view, size);
        assert!(after.stops < goal - 3.0, "{after:?}");
        // A fixed exposure is written as it is, with no metering.
        let fixed = rig.frame(&Wanted { compensation: -1.25, ..Default::default() }, &dark, size);
        assert_eq!((fixed.stops, fixed.valid), (-1.25, 1.0));
    }

    #[test]
    fn the_histogram_range_is_the_frames_to_choose() {
        let Some(mut rig) = Rig::new() else { return };
        let size = (16, 16);
        // 2^-2 in a range of -4..4: a bin of 1/32 stop, bin 56 of 256 holds -2.25..-2.22.
        let view = rig.view(size, &vec![grey(-2.0 + 1.0 / 64.0); 256]);
        let wanted = Wanted { range_low: -4.0, range_high: 4.0, ..auto() };
        let state = rig.frame(&wanted, &view, size);
        let bins = reference::histogram(&held(&vec![grey(-2.0 + 1.0 / 64.0); 256]), 16, 16, &wanted);
        let gpu: Vec<u64> = rig.histogram().into_iter().map(u64::from).collect();
        assert_eq!(gpu, bins);
        assert!((state.average - reference::average(&bins, &wanted).unwrap()).abs() < 1e-4);
        // What lies outside falls in the first and last bins.
        let wanted = Wanted { range_low: -1.0, range_high: 1.0, ..auto() };
        rig.frame(&wanted, &view, size);
        assert_eq!(rig.histogram()[0], 256 * 256);
    }
}
