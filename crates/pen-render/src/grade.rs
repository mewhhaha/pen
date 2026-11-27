// The color grade's inputs. The grade itself is WGSL (`shaders/grade.wgsl`, run by
// the present pass); this is what the frame's settings become: the uniform it
// reads, and the look-up tables, which the frame names by material.
//
// A table is a strip texture: N slices of N by N texels side by side (N from 2 to
// 64, so N * N wide and N tall), blue picking the slice, red running along it,
// green down; texel (r, g, b) of an identity table holds (r, g, b) / (N - 1). It
// holds colors in the grade's own log encoding (ACEScc), where 0 is a thousandth of
// SDR white and 1 more than two hundred times it. It is read as data, not as a
// color picture (a texture `kind` of data, or one that is not sRGB), at its full
// size; until it has streamed in the grade goes without it.
use bytemuck::{Pod, Zeroable};
use pen_protocol::Grade;

/// A table's finest level, as the present pass reads it.
#[derive(Clone)]
pub(crate) struct Lut {
    pub view: wgpu::TextureView,
    /// Which view it is: a table that streams or reloads gets a new one.
    pub id: u64,
    pub size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GradeUniform {
    basics: [f32; 4],
    tone: [f32; 4],
    lift: [f32; 4],
    gamma: [f32; 4],
    gain: [f32; 4],
    luts: [f32; 4],
    vignette: [f32; 4],
}

/// Whether the grade, leaving the tables aside, changes nothing.
pub(crate) fn neutral(grade: &Grade) -> bool {
    let (lift, gamma, gain) = (&grade.lift, &grade.gamma, &grade.gain);
    grade.temperature == 0.0
        && grade.tint == 0.0
        && grade.exposure == 0.0
        && grade.contrast == 1.0
        && grade.saturation == 1.0
        && grade.vibrance == 0.0
        && [lift.x, lift.y, lift.z] == [0.0; 3]
        && [gamma.x, gamma.y, gamma.z] == [1.0; 3]
        && [gain.x, gain.y, gain.z] == [1.0; 3]
        && grade.vignette == 0.0
}

/// The tables the grade uses, in order: the first alone if only one is there,
/// then the blend.
pub(crate) fn tables(grade: &Grade, a: Option<Lut>, b: Option<Lut>) -> [Option<Lut>; 2] {
    let used = if grade.lut_weight > 0.0 { [a, b] } else { [None, None] };
    match used {
        [None, second] => [second, None],
        both => both,
    }
}

/// The uniform for `grade` with these `tables`; its last lane of `luts` is 1 when
/// the present pass has anything to do.
#[cfg(test)]
pub(crate) fn uniform(grade: &Grade, tables: &[Option<Lut>; 2]) -> GradeUniform {
    uniform_with_tonemap(grade, tables, crate::tonemap::TonemapOperator::Shoulder, 1.0)
}

/// The uniform for `grade` including display tonemap operator and encoded HDR peak.
pub(crate) fn uniform_with_tonemap(
    grade: &Grade,
    tables: &[Option<Lut>; 2],
    tonemap: crate::tonemap::TonemapOperator,
    peak_encoded: f32,
) -> GradeUniform {
    let size = |table: &Option<Lut>| table.as_ref().map_or(0.0, |table| table.size as f32);
    let active = !neutral(grade) || tables[0].is_some();
    GradeUniform {
        basics: [grade.temperature, grade.tint, grade.exposure, grade.contrast],
        tone: [grade.pivot, grade.saturation, grade.vibrance, grade.lut_weight],
        lift: [grade.lift.x, grade.lift.y, grade.lift.z, 0.0],
        gamma: [grade.gamma.x, grade.gamma.y, grade.gamma.z, 0.0],
        gain: [grade.gain.x, grade.gain.y, grade.gain.z, 0.0],
        luts: [
            size(&tables[0]),
            size(&tables[1]),
            if tables[1].is_some() { grade.lut_blend } else { 0.0 },
            if active { 1.0 } else { 0.0 },
        ],
        vignette: [grade.vignette, grade.vignette_radius, (tonemap as u32) as f32, peak_encoded.max(1.0)],
    }
}

/// The grade in f64, term for term as `shaders/grade.wgsl` and Ink's `color.grade`
/// say it: what the shader is checked against.
#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test pictures, bins and stops"
)]
pub(crate) mod reference {
    pub fn srgb_decode(value: f64) -> f64 {
        let c = value.max(0.0);
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    }

    pub fn srgb_encode(value: f64) -> f64 {
        let l = value.max(0.0);
        if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 }
    }

    pub fn luminance(light: [f64; 3]) -> f64 {
        0.2126 * light[0] + 0.7152 * light[1] + 0.0722 * light[2]
    }

    pub const CC_LOW: f64 = (-16.0 + 9.72) / 17.52;

    pub fn cc_high() -> f64 {
        (65504f64.log2() + 9.72) / 17.52
    }

    pub fn cc_encode(x: f64) -> f64 {
        if x <= 0.0 {
            CC_LOW
        } else if x < 2f64.powi(-15) {
            ((2f64.powi(-16) + x * 0.5).log2() + 9.72) / 17.52
        } else {
            (x.log2() + 9.72) / 17.52
        }
    }

    pub fn cc_decode(v: f64) -> f64 {
        if v < (9.72 - 15.0) / 17.52 {
            ((2f64.powf(v * 17.52 - 9.72) - 2f64.powi(-16)) * 2.0).max(0.0)
        } else {
            2f64.powf(v * 17.52 - 9.72).min(65504.0)
        }
    }

    /// The grade's settings in f64, with the table weights.
    #[derive(Clone, Copy, Debug)]
    pub struct Settings {
        pub temperature: f64,
        pub tint: f64,
        pub exposure: f64,
        pub contrast: f64,
        pub pivot: f64,
        pub saturation: f64,
        pub vibrance: f64,
        pub lift: [f64; 3],
        pub gamma: [f64; 3],
        pub gain: [f64; 3],
        pub vignette: f64,
        pub radius: f64,
        pub blend: f64,
        pub weight: f64,
    }

    impl Settings {
        pub const NEUTRAL: Settings = Settings {
            temperature: 0.0,
            tint: 0.0,
            exposure: 0.0,
            contrast: 1.0,
            pivot: 0.18,
            saturation: 1.0,
            vibrance: 0.0,
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
            vignette: 0.0,
            radius: 0.5,
            blend: 0.0,
            weight: 1.0,
        };
    }

    /// A look-up table's texels as the strip holds them, by (blue * size + red, green).
    #[derive(Clone, Debug)]
    pub struct Table {
        pub size: usize,
        pub texels: Vec<[f64; 3]>,
    }

    impl Table {
        pub fn at(&self, r: usize, g: usize, b: usize) -> [f64; 3] {
            self.texels[g * self.size * self.size + b * self.size + r]
        }

        /// Trilinear, texel centres at i / (size - 1).
        pub fn lookup(&self, v: [f64; 3]) -> [f64; 3] {
            let top = (self.size - 1) as f64;
            let p = v.map(|value| value.clamp(0.0, 1.0) * top);
            let base = p.map(f64::floor);
            let f = [p[0] - base[0], p[1] - base[1], p[2] - base[2]];
            let low = base.map(|value| (value as usize).min(self.size - 1));
            let high = low.map(|value| (value + 1).min(self.size - 1));
            let mut out = [0.0; 3];
            for (corner, weight) in (0..8).map(|corner| {
                let pick = |bit: usize, axis: usize| {
                    if corner >> bit & 1 == 1 { (high[axis], f[axis]) } else { (low[axis], 1.0 - f[axis]) }
                };
                let ((r, wr), (g, wg), (b, wb)) = (pick(0, 0), pick(1, 1), pick(2, 2));
                (self.at(r, g, b), wr * wg * wb)
            }) {
                for channel in 0..3 {
                    out[channel] += corner[channel] * weight;
                }
            }
            out
        }
    }

    pub fn vignette_factor(strength: f64, radius: f64, uv: [f64; 2]) -> f64 {
        let radius = radius.min(0.999);
        let d = ((uv[0] - 0.5).powi(2) + (uv[1] - 0.5).powi(2)).sqrt() * 2f64.sqrt();
        let t = ((d - radius) / (1.0 - radius)).clamp(0.0, 1.0);
        1.0 - strength * t * t * (3.0 - 2.0 * t)
    }

    /// Linear light `light` at `uv`, graded, with these tables (the first alone, or
    /// blended with the second by `blend`).
    pub fn grade(s: &Settings, tables: [Option<&Table>; 2], light: [f64; 3], uv: [f64; 2]) -> [f64; 3] {
        let factor = vignette_factor(s.vignette, s.radius, uv);
        let (t, tint) = (s.temperature, s.tint);
        let stops = [0.5 * t + 0.25 * tint, -0.5 * tint, -0.5 * t + 0.25 * tint];
        let mut c = [0.0; 3];
        for i in 0..3 {
            c[i] = light[i].max(0.0) * factor * 2f64.powf(stops[i] + s.exposure);
        }
        let pivot = cc_encode(s.pivot);
        for value in &mut c {
            *value = cc_decode(((cc_encode(*value) - pivot) * s.contrast + pivot).clamp(CC_LOW, cc_high()));
        }
        for (i, value) in c.iter_mut().enumerate() {
            *value = (*value * s.gain[i] + s.lift[i]).max(0.0).powf(1.0 / s.gamma[i]);
        }
        let y = luminance(c);
        let high = c[0].max(c[1]).max(c[2]);
        let low = c[0].min(c[1]).min(c[2]);
        let chroma = if high > 1e-6 { (high - low) / high } else { 0.0 };
        let weight = if s.vibrance > 0.0 { 1.0 - chroma } else { chroma };
        let k = s.saturation * (1.0 + s.vibrance * weight);
        for value in &mut c {
            *value = (*value * k + y * (1.0 - k)).max(0.0);
        }
        if let (Some(a), true) = (tables[0], s.weight > 0.0) {
            let v = c.map(cc_encode);
            let inside = v.map(|value| value.clamp(0.0, 1.0));
            let mut looked = a.lookup(inside);
            if let Some(b) = tables[1]
                && s.blend > 0.0
            {
                let other = b.lookup(inside);
                for i in 0..3 {
                    looked[i] += (other[i] - looked[i]) * s.blend;
                }
            }
            for ((c, v), (looked, inside)) in c.iter_mut().zip(v).zip(looked.into_iter().zip(inside)) {
                *c = cc_decode(v + (looked - inside) * s.weight);
            }
        }
        c
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
    use wgpu::util::DeviceExt;

    use super::reference::{self, Settings, Table};
    use super::tables as tables_of;
    use super::*;
    use crate::post::{GRADE_WGSL, TRANSFER_WGSL};
    use crate::testing::{device, read};

    const HARNESS: &str = "
        @group(0) @binding(0) var<uniform> g: Grade;
        @group(0) @binding(1) var<storage, read> inputs: array<vec4f>;
        @group(0) @binding(2) var<storage, read_write> outputs: array<vec4f>;
        @group(0) @binding(3) var lut_a: texture_2d<f32>;
        @group(0) @binding(4) var lut_b: texture_2d<f32>;
        // Two vec4s a sample: its light and u, then v. The unused lane of `gain`
        // asks for the plain table lookup of the first table instead of the grade.
        @compute @workgroup_size(64) fn run(@builtin(global_invocation_id) id: vec3u) {
          let i = id.x;
          if (i >= arrayLength(&inputs) / 2u) { return; }
          let a = inputs[2u * i];
          let b = inputs[2u * i + 1u];
          if (g.gain.w > 0.5) {
            outputs[i] = vec4f(lut_lookup(lut_a, i32(g.luts.x), a.xyz), 0.0);
          } else {
            outputs[i] = vec4f(grade_apply(g, lut_a, lut_b, a.xyz, vec2f(a.w, b.x)), 0.0);
          }
        }";

    struct Rig {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
    }

    /// A deterministic stream of numbers in 0..1.
    struct Noise(u64);

    impl Noise {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    impl Rig {
        fn new() -> Option<Rig> {
            let (device, queue) = device()?;
            let code = format!("{TRANSFER_WGSL}\n{GRADE_WGSL}\n{HARNESS}");
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("grade test"),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("grade test"),
                layout: None,
                module: &module,
                entry_point: Some("run"),
                compilation_options: Default::default(),
                cache: crate::pipeline_cache::active().as_ref(),
            });
            Some(Rig { device, queue, pipeline })
        }

        /// A strip texture of `table`, its texels as 8-bit unorm like an authored one.
        fn strip(&self, table: &Table) -> Lut {
            let size = table.size as u32;
            let bytes: Vec<u8> = table
                .texels
                .iter()
                .flat_map(|texel| [texel[0], texel[1], texel[2], 1.0].map(|value| (value * 255.0).round() as u8))
                .collect();
            let texture = self.device.create_texture_with_data(
                &self.queue,
                &wgpu::TextureDescriptor {
                    label: Some("table"),
                    size: wgpu::Extent3d { width: size * size, height: size, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &bytes,
            );
            Lut { view: texture.create_view(&Default::default()), id: 1, size }
        }

        /// The shader's answer for each of `samples` (linear light, then uv).
        fn run(
            &self,
            uniform: &GradeUniform,
            tables: &[Option<Lut>; 2],
            samples: &[([f64; 3], [f64; 2])],
        ) -> Vec<[f64; 3]> {
            let blank = self.strip(&Table { size: 2, texels: vec![[0.0; 3]; 8] });
            let inputs: Vec<f32> = samples
                .iter()
                .flat_map(|(light, uv)| {
                    [light[0] as f32, light[1] as f32, light[2] as f32, uv[0] as f32, uv[1] as f32, 0.0, 0.0, 0.0]
                })
                .collect();
            let storage = |contents: &[u8], usage: wgpu::BufferUsages| {
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents, usage })
            };
            let parameters = storage(bytemuck::bytes_of(uniform), wgpu::BufferUsages::UNIFORM);
            let input = storage(bytemuck::cast_slice(&inputs), wgpu::BufferUsages::STORAGE);
            let size = (samples.len() * 16) as u64;
            let output = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let (a, b) = (tables[0].as_ref().unwrap_or(&blank), tables[1].as_ref().unwrap_or(&blank));
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: parameters.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&a.view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&b.view) },
                ],
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(samples.len().div_ceil(64) as u32, 1, 1);
            }
            self.queue.submit([encoder.finish()]);
            let bytes = read(&self.device, &self.queue, &output, size);
            let floats: &[f32] = bytemuck::cast_slice(&bytes);
            floats.chunks(4).map(|texel| [f64::from(texel[0]), f64::from(texel[1]), f64::from(texel[2])]).collect()
        }
    }

    /// Light spread over the stops that matter, in each channel, with black and the
    /// extremes.
    fn lights(count: usize, seed: u64) -> Vec<([f64; 3], [f64; 2])> {
        let mut noise = Noise(seed);
        let mut out: Vec<([f64; 3], [f64; 2])> = vec![
            ([0.0; 3], [0.5, 0.5]),
            ([1.0; 3], [0.0, 0.0]),
            ([0.18; 3], [1.0, 1.0]),
            ([1e-6, 1e-4, 1e-3], [0.25, 0.75]),
            ([60000.0, 3.0, 0.0], [0.9, 0.1]),
        ];
        while out.len() < count {
            // A luminance, and three channels within three stops of it.
            let base = 2f64.powf(noise.next() * 22.0 - 14.0);
            let channel = |noise: &mut Noise| base * 2f64.powf(noise.next() * 6.0 - 3.0);
            let light = [channel(&mut noise), channel(&mut noise), channel(&mut noise)];
            out.push((light, [noise.next(), noise.next()]));
        }
        out
    }

    /// Whether `got` is `want` to a relative error, an absolute one, and what a float32 loses to the
    /// brightest channel beside a dim one.
    fn close(got: &[f64; 3], want: &[f64; 3], relative: f64, absolute: f64) -> bool {
        let brightest = want.iter().fold(0.0f64, |most, w| most.max(w.abs()));
        got.iter().zip(want).all(|(g, w)| (g - w).abs() <= relative * w.abs() + absolute + 4e-7 * brightest)
    }

    fn grade_of(s: &Settings) -> pen_protocol::Grade {
        let vector = |v: [f64; 3]| pen_protocol::Vector { x: v[0] as f32, y: v[1] as f32, z: v[2] as f32 };
        pen_protocol::Grade {
            temperature: s.temperature as f32,
            tint: s.tint as f32,
            exposure: s.exposure as f32,
            contrast: s.contrast as f32,
            pivot: s.pivot as f32,
            saturation: s.saturation as f32,
            vibrance: s.vibrance as f32,
            lift: vector(s.lift),
            gamma: vector(s.gamma),
            gain: vector(s.gain),
            vignette: s.vignette as f32,
            vignette_radius: s.radius as f32,
            lut_blend: s.blend as f32,
            lut_weight: s.weight as f32,
            ..Default::default()
        }
    }

    /// An identity table of `size`, as 8-bit texels.
    fn identity(size: usize) -> Table {
        let level = |i: usize| (i as f64 / (size - 1) as f64 * 255.0).round() / 255.0;
        let mut texels = vec![[0.0; 3]; size * size * size];
        for g in 0..size {
            for b in 0..size {
                for r in 0..size {
                    texels[g * size * size + b * size + r] = [level(r), level(g), level(b)];
                }
            }
        }
        Table { size, texels }
    }

    /// A table of arbitrary 8-bit colors.
    fn scrambled(size: usize, seed: u64) -> Table {
        let mut noise = Noise(seed);
        let texels =
            (0..size * size * size).map(|_| [(); 3].map(|()| (noise.next() * 255.0).round() / 255.0)).collect();
        Table { size, texels }
    }

    #[test]
    fn the_uniform_says_when_the_grade_applies() {
        let none = [None, None];
        let flat = pen_protocol::Grade::default();
        assert!(neutral(&flat));
        assert_eq!(uniform(&flat, &none).luts[3], 0.0);
        let warm = pen_protocol::Grade { temperature: 0.2, ..flat };
        assert!(!neutral(&warm));
        assert_eq!(uniform(&warm, &none).luts[3], 1.0);
        let vignette = pen_protocol::Grade { vignette: 0.1, ..flat };
        assert_eq!(uniform(&vignette, &none).luts[3], 1.0);
    }

    #[test]
    fn a_default_frame_grades_nothing() {
        // The Rust defaults are the schema's: a frame that leaves the grade alone is
        // the identity, with no table.
        let grade = pen_protocol::Grade::default();
        assert!(neutral(&grade));
        assert_eq!((grade.contrast, grade.saturation, grade.pivot, grade.lut_weight), (1.0, 1.0, 0.18, 1.0));
        assert_eq!((grade.lut_a, grade.lut_b), (0, 0));
        let exposure = pen_protocol::Exposure::default();
        assert!(!exposure.auto);
        assert_eq!((exposure.compensation, exposure.min, exposure.max, exposure.target), (0.0, -6.0, 6.0, 0.18));
    }

    #[test]
    fn the_identity_grade_is_the_identity() {
        let Some(rig) = Rig::new() else { return };
        let uniform = uniform(&grade_of(&Settings::NEUTRAL), &[None, None]);
        let samples = lights(4000, 7);
        let got = rig.run(&uniform, &[None, None], &samples);
        for ((light, _), got) in samples.iter().zip(&got) {
            // Black stays black, and the rest round trips the log curve within float.
            assert!(close(got, light, 5e-5, 1e-7), "{light:?} became {got:?}");
        }
    }

    #[test]
    fn the_shader_matches_the_f64_reference_for_every_setting() {
        let Some(rig) = Rig::new() else { return };
        let neutral = Settings::NEUTRAL;
        let settings = [
            Settings { temperature: 0.7, ..neutral },
            Settings { temperature: -1.0, tint: 0.4, ..neutral },
            Settings { tint: -0.8, ..neutral },
            Settings { exposure: 1.5, ..neutral },
            Settings { exposure: -3.0, ..neutral },
            Settings { contrast: 1.6, ..neutral },
            Settings { contrast: 0.4, pivot: 0.5, ..neutral },
            Settings { contrast: 3.9, pivot: 0.02, ..neutral },
            Settings { saturation: 0.0, ..neutral },
            Settings { saturation: 2.5, ..neutral },
            Settings { vibrance: 0.8, ..neutral },
            Settings { vibrance: -0.9, ..neutral },
            Settings { lift: [0.05, -0.02, 0.1], ..neutral },
            Settings { gamma: [0.6, 1.0, 2.2], ..neutral },
            Settings { gain: [1.4, 0.8, 2.0], ..neutral },
            Settings { vignette: 0.7, radius: 0.3, ..neutral },
            Settings {
                temperature: 0.3,
                tint: -0.2,
                exposure: 0.5,
                contrast: 1.2,
                pivot: 0.3,
                saturation: 1.3,
                vibrance: 0.4,
                lift: [0.01, 0.02, -0.01],
                gamma: [1.1, 0.9, 1.3],
                gain: [1.1, 0.95, 1.2],
                vignette: 0.4,
                radius: 0.5,
                ..neutral
            },
        ];
        let samples = lights(1500, 11);
        for s in settings {
            let got = rig.run(&uniform(&grade_of(&s), &[None, None]), &[None, None], &samples);
            for ((light, uv), got) in samples.iter().zip(&got) {
                let want = reference::grade(&s, [None, None], *light, *uv);
                assert!(close(got, &want, 3e-4, 1e-5), "{s:?} on {light:?} at {uv:?}: {got:?}, want {want:?}");
            }
        }
    }

    #[test]
    fn white_balance_and_the_cdl_move_what_they_say() {
        // Warmer is redder and bluer-less; lift lifts black; gain scales white.
        let neutral = Settings::NEUTRAL;
        let grey = [0.18; 3];
        let warm = reference::grade(&Settings { temperature: 1.0, ..neutral }, [None, None], grey, [0.5, 0.5]);
        assert!(warm[0] > grey[0] && warm[2] < grey[2]);
        let lifted = reference::grade(&Settings { lift: [0.1; 3], ..neutral }, [None, None], [0.0; 3], [0.5, 0.5]);
        assert!(lifted.iter().all(|&value| (value - 0.1).abs() < 1e-9), "{lifted:?}");
        let gained =
            reference::grade(&Settings { gain: [2.0, 1.0, 0.5], ..neutral }, [None, None], [0.5; 3], [0.5, 0.5]);
        assert!((gained[0] - 1.0).abs() < 1e-6 && (gained[2] - 0.25).abs() < 1e-6, "{gained:?}");
        // Contrast turns about the pivot: the pivot itself stays.
        let pivot = reference::grade(&Settings { contrast: 2.0, ..neutral }, [None, None], [0.18; 3], [0.5, 0.5]);
        assert!(close(&pivot, &[0.18; 3], 1e-9, 1e-12), "{pivot:?}");
        // Saturation 0 is the luminance.
        let colour = [0.6, 0.2, 0.1];
        let grey = reference::grade(&Settings { saturation: 0.0, ..neutral }, [None, None], colour, [0.5, 0.5]);
        assert!(grey.iter().all(|&value| (value - reference::luminance(colour)).abs() < 1e-9));
        // The vignette leaves the middle and darkens a corner to 1 - strength.
        let centre = reference::vignette_factor(0.6, 0.5, [0.5, 0.5]);
        let corner = reference::vignette_factor(0.6, 0.5, [0.0, 1.0]);
        assert_eq!(centre, 1.0);
        assert!((corner - 0.4).abs() < 1e-12);
    }

    #[test]
    fn table_lookups_match_trilinear_sampling_on_the_cpu() {
        let Some(rig) = Rig::new() else { return };
        for (size, seed) in [(2usize, 3u64), (5, 5), (17, 9), (32, 13), (64, 17)] {
            let table = scrambled(size, seed);
            let lut = rig.strip(&table);
            let mut noise = Noise(seed * 101);
            // Inside, on the grid, on the edges and beyond.
            let mut samples: Vec<([f64; 3], [f64; 2])> = vec![
                ([0.0; 3], [0.0; 2]),
                ([1.0; 3], [0.0; 2]),
                ([0.5; 3], [0.0; 2]),
                ([-0.3, 1.7, 0.5], [0.0; 2]),
                ([1.0 / (size - 1) as f64; 3], [0.0; 2]),
            ];
            samples.extend((0..800).map(|_| ([noise.next(), noise.next(), noise.next()], [0.0; 2])));
            let mut uniform = uniform(&grade_of(&Settings::NEUTRAL), &[Some(lut.clone()), None]);
            // The lookup alone is asked for in `gain`'s unused fourth lane.
            uniform.gain[3] = 1.0;
            let got = rig.run(&uniform, &[Some(lut), None], &samples);
            for ((v, _), got) in samples.iter().zip(&got) {
                let want = table.lookup(*v);
                assert!(close(got, &want, 0.0, 1e-5), "size {size} at {v:?}: {got:?}, want {want:?}");
            }
        }
    }

    #[test]
    fn the_identity_table_changes_nothing_but_the_quantizing() {
        let Some(rig) = Rig::new() else { return };
        for size in [2usize, 8, 32] {
            let table = identity(size);
            let tables = [Some(rig.strip(&table)), None];
            let samples = lights(1500, 23 + size as u64);
            let graded = Settings::NEUTRAL;
            let got = rig.run(&uniform(&grade_of(&graded), &tables), &tables, &samples);
            for ((light, uv), got) in samples.iter().zip(&got) {
                // The table's own steps of 1/255 of the log encoding bound the change.
                let want = reference::grade(&graded, [Some(&table), None], *light, *uv);
                assert!(close(got, &want, 3e-4, 1e-5), "size {size}: {light:?} gave {got:?}, want {want:?}");
                let before = light.map(reference::cc_encode);
                let after = got.map(reference::cc_encode);
                for channel in 0..3 {
                    if (0.0..=1.0).contains(&before[channel]) && size > 2 {
                        assert!((after[channel] - before[channel]).abs() <= 1.0 / 255.0 + 1e-4);
                    }
                }
            }
        }
    }

    #[test]
    fn tables_blend_and_weigh_as_the_reference_says() {
        let Some(rig) = Rig::new() else { return };
        let (a, b) = (scrambled(8, 31), scrambled(8, 37));
        let tables = [Some(rig.strip(&a)), Some(rig.strip(&b))];
        let samples = lights(600, 41);
        let grade = Settings { saturation: 1.2, temperature: 0.1, ..Settings::NEUTRAL };
        for (blend, weight) in [(0.0, 1.0), (1.0, 1.0), (0.5, 1.0), (0.3, 0.6), (0.7, 0.0), (0.5, 0.25)] {
            let s = Settings { blend, weight, ..grade };
            let got = rig.run(&uniform(&grade_of(&s), &tables), &tables, &samples);
            for ((light, uv), got) in samples.iter().zip(&got) {
                let want = reference::grade(&s, [Some(&a), Some(&b)], *light, *uv);
                assert!(
                    close(got, &want, 3e-4, 1e-5),
                    "blend {blend} weight {weight}: {light:?} gave {got:?}, want {want:?}"
                );
            }
        }
        // The blend's ends are each table alone, and no weight is no table.
        let alone = |blend: f64, which: &Table| {
            let s = Settings { blend, weight: 1.0, ..grade };
            samples.iter().map(|(light, uv)| reference::grade(&s, [Some(which), None], *light, *uv)).collect::<Vec<_>>()
        };
        let first = rig.run(&uniform(&grade_of(&Settings { blend: 0.0, ..grade }), &tables), &tables, &samples);
        for (got, want) in first.iter().zip(alone(0.0, &a)) {
            assert!(close(got, &want, 3e-4, 1e-5));
        }
        let second = rig.run(&uniform(&grade_of(&Settings { blend: 1.0, ..grade }), &tables), &tables, &samples);
        for (got, want) in second.iter().zip(alone(0.0, &b)) {
            assert!(close(got, &want, 3e-4, 1e-5));
        }
        // A table alone applies; the second stands in for a missing first; none at
        // all, or no weight, leaves the grade to the rest.
        let lone = tables_of(&grade_of(&grade), Some(tables[0].clone().unwrap()), None);
        assert_eq!((lone[0].is_some(), lone[1].is_some()), (true, false));
        let second = tables_of(&grade_of(&grade), None, Some(tables[1].clone().unwrap()));
        assert_eq!((second[0].as_ref().map(|table| table.size), second[1].is_some()), (Some(8), false));
        assert_eq!(uniform(&grade_of(&grade), &lone).luts[3], 1.0);
        let weightless =
            tables_of(&grade_of(&Settings { weight: 0.0, ..grade }), Some(tables[0].clone().unwrap()), None);
        assert!(weightless[0].is_none());
        let none = [None, None];
        let off = Settings { weight: 0.0, ..grade };
        let got = rig.run(&uniform(&grade_of(&off), &tables), &tables, &samples);
        let bare = rig.run(&uniform(&grade_of(&grade), &none), &none, &samples);
        for (got, want) in got.iter().zip(&bare) {
            assert!(close(got, want, 1e-6, 1e-9));
        }
    }
}
