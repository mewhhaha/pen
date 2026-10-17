// Clustered forward lighting for the frame's point and spot lights (the
// render packet's lights section). The view is cut into clusters: 16 x 9
// tiles of the picture, and 24 slices of the depth ahead, closer together
// near the eye (evenly spaced for an orthographic camera). Each light's reach,
// a sphere (a spot's cone in the smallest sphere round it that this finds),
// goes into every cluster it may touch, and the shaders of lit materials read
// only their fragment's cluster's lights: `LIGHTS_WGSL`, bound as group 2 of
// every mesh pipeline. A frame without lights costs one small uniform write.
use anyhow::{Result, ensure};
use pen_protocol::{Light, Matrix, RenderFrame};

/// Tiles across and down the picture, and slices of depth.
pub(crate) const TILES: (u32, u32) = (16, 9);
pub(crate) const SLICES: u32 = 24;
const CLUSTERS: usize = 16 * 9 * 24;

/// The most lights one cluster takes; past them a cluster's lights are left
/// out, in the frame's order.
const PER_CLUSTER: u32 = 256;

/// WGSL a material shader adds to light with the frame's lights:
/// `clustered_light(world, normal, position, view_z)`.
pub const LIGHTS_WGSL: &str = include_str!("shaders/lights.wgsl");

/// How the depth ahead is cut into slices: a depth's slice is
/// `log(depth) * scale + bias`, or `depth * scale + bias` when `linear`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Slicing {
    pub near: f64,
    pub far: f64,
    pub linear: bool,
    pub scale: f64,
    pub bias: f64,
}

impl Slicing {
    fn slice(&self, depth: f64) -> u32 {
        let along = if self.linear { depth } else { depth.max(1e-6).ln() };
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped to the slices")]
        let slice = (along * self.scale + self.bias).floor().clamp(0.0, f64::from(SLICES - 1)) as u32;
        slice
    }
}

/// Each cluster's lights: its first index into `indices` and how many.
pub(crate) struct Assignment {
    pub ranges: Vec<[u32; 2]>,
    pub indices: Vec<u32>,
    pub slicing: Slicing,
}

fn transform(m: &Matrix, point: [f64; 4]) -> [f64; 4] {
    let mut out = [0.0; 4];
    for (row, value) in out.iter_mut().enumerate() {
        *value = (0..4).map(|k| f64::from(m[k * 4 + row]) * point[k]).sum();
    }
    out
}

/// The slicing of the depth `projection` sees, as far as `farthest` when it
/// sees forever.
fn slicing(projection: &Matrix, farthest: f64) -> Slicing {
    let inverse = crate::matrix::invert(projection);
    // The depth ahead of the eye at a depth-buffer value.
    let depth = |value: f64| {
        let point = transform(&inverse, [0.0, 0.0, value, 1.0]);
        if point[3].abs() < 1e-12 { f64::INFINITY } else { -point[2] / point[3] }
    };
    let near = depth(0.0).max(1e-4);
    let mut far = depth(1.0);
    if !far.is_finite() || far <= near {
        far = farthest.max(near * 2.0);
    }
    let linear = projection[11] == 0.0;
    let slices = f64::from(SLICES);
    let (scale, bias) = if linear {
        (slices / (far - near), -near * slices / (far - near))
    } else {
        let scale = slices / (far / near).ln();
        (scale, -near.ln() * scale)
    };
    Slicing { near, far, linear, scale, bias }
}

/// The sphere round what `light` reaches, in view space.
fn reach(light: &Light, view: &Matrix) -> ([f64; 3], f64) {
    let position = [f64::from(light.position.x), f64::from(light.position.y), f64::from(light.position.z)];
    let radius = f64::from(light.radius);
    let outer = f64::from(light.outer);
    let direction = [f64::from(light.direction.x), f64::from(light.direction.y), f64::from(light.direction.z)];
    let length = direction.iter().map(|value| value * value).sum::<f64>().sqrt();
    // A spot's cone, its cap included, fits a smaller sphere along its axis.
    let (center, radius) = if outer > 0.0 && length > 0.0 {
        let along = |distance: f64| std::array::from_fn(|axis| position[axis] + direction[axis] / length * distance);
        if outer >= std::f64::consts::FRAC_1_SQRT_2 {
            let half = radius / (2.0 * outer);
            (along(half), half)
        } else {
            (along(radius * outer), radius * (1.0 - outer * outer).sqrt())
        }
    } else {
        (position, radius)
    };
    let seen = transform(view, [center[0], center[1], center[2], 1.0]);
    ([seen[0], seen[1], seen[2]], radius)
}

/// The clusters each of `lights` may light, seen through `view` and
/// `projection`.
pub(crate) fn assign(lights: &[Light], view: &Matrix, projection: &Matrix) -> Assignment {
    let spheres: Vec<([f64; 3], f64)> = lights.iter().map(|light| reach(light, view)).collect();
    let farthest = spheres.iter().map(|(center, radius)| radius - center[2]).fold(1.0, f64::max);
    let slicing = slicing(projection, farthest);
    // Each light's box of clusters, if it reaches any.
    let boxes: Vec<Option<[u32; 6]>> = spheres
        .iter()
        .map(|&(center, radius)| {
            let (closest, farthest) = (-(center[2] + radius), -(center[2] - radius));
            if farthest < slicing.near || closest > slicing.far {
                return None;
            }
            // The corners of its box, the near ones no closer than the near
            // plane, on the picture.
            let near_z = -closest.max(slicing.near);
            let far_z = center[2] - radius;
            let (mut low, mut high) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
            for corner in 0..8 {
                let x = center[0] + if corner & 1 == 0 { -radius } else { radius };
                let y = center[1] + if corner & 2 == 0 { -radius } else { radius };
                let z = if corner & 4 == 0 { near_z } else { far_z };
                let clip = transform(projection, [x, y, z, 1.0]);
                let ndc = [clip[0] / clip[3], clip[1] / clip[3]];
                for axis in 0..2 {
                    low[axis] = low[axis].min(ndc[axis]);
                    high[axis] = high[axis].max(ndc[axis]);
                }
            }
            if high[0] < -1.0 || low[0] > 1.0 || high[1] < -1.0 || low[1] > 1.0 {
                return None;
            }
            // As the shaders find a fragment's tile; picture rows go down
            // while clip-space y goes up.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped to the tiles")]
            let tile =
                |along: f64, tiles: u32| (along * f64::from(tiles)).floor().clamp(0.0, f64::from(tiles - 1)) as u32;
            Some([
                tile(low[0] * 0.5 + 0.5, TILES.0),
                tile(high[0] * 0.5 + 0.5, TILES.0),
                tile(0.5 - high[1] * 0.5, TILES.1),
                tile(0.5 - low[1] * 0.5, TILES.1),
                slicing.slice(closest.max(slicing.near)),
                slicing.slice(farthest.min(slicing.far)),
            ])
        })
        .collect();
    let clusters_of = |[x0, x1, y0, y1, z0, z1]: [u32; 6]| {
        (z0..=z1).flat_map(move |z| {
            (y0..=y1).flat_map(move |y| (x0..=x1).map(move |x| ((z * TILES.1 + y) * TILES.0 + x) as usize))
        })
    };
    let mut counts = vec![0u32; CLUSTERS];
    for cluster in boxes.iter().flatten().flat_map(|&bounds| clusters_of(bounds)) {
        counts[cluster] = (counts[cluster] + 1).min(PER_CLUSTER);
    }
    let mut ranges = Vec::with_capacity(CLUSTERS);
    let mut first = 0;
    for &count in &counts {
        ranges.push([first, 0]);
        first += count;
    }
    let mut indices = vec![0; usize::try_from(first).unwrap_or(0)];
    for (light, bounds) in (0u32..).zip(&boxes) {
        let Some(bounds) = *bounds else { continue };
        for cluster in clusters_of(bounds) {
            let [start, taken] = &mut ranges[cluster];
            if *taken < counts[cluster] {
                indices[(*start + *taken) as usize] = light;
                *taken += 1;
            }
        }
    }
    Assignment { ranges, indices, slicing }
}

/// What a frame's lights must be.
pub(crate) fn validate(lights: &[Light]) -> Result<()> {
    for light in lights {
        ensure!(light.radius > 0.0, "game returned a light without reach");
        ensure!(
            (-1.0..=1.0).contains(&light.outer) && (-1.0..=1.0).contains(&light.inner),
            "game returned a light whose cone is not a cosine"
        );
    }
    Ok(())
}

/// The layout of group 2 of every mesh pipeline: the grid, the lights, each
/// cluster's range and the lights' indices.
pub(crate) fn group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    // A WebGL2-class device has no storage buffers: the grid alone, of no
    // lights, which shaders reading the lights cannot draw with there.
    let limited = crate::downlevel::limited(device);
    let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lights"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(GRID_BYTES),
                },
                count: None,
            },
            storage(1),
            storage(2),
            storage(3),
        ][..if limited { 1 } else { 4 }],
    })
}

const GRID_BYTES: u64 = 48;
const LIGHT_FLOATS: usize = 12;

/// The grid's uniform: its counts, its scales and whether its slices are
/// linear.
fn grid(lights: u32, slicing: &Slicing, size: (u32, u32)) -> [u32; 12] {
    let float = |value: f64| crate::matrix::narrow(value).to_bits();
    [
        TILES.0,
        TILES.1,
        SLICES,
        lights,
        float(f64::from(TILES.0) / f64::from(size.0.max(1))),
        float(f64::from(TILES.1) / f64::from(size.1.max(1))),
        float(slicing.scale),
        float(slicing.bias),
        float(if slicing.linear { 1.0 } else { 0.0 }),
        0,
        0,
        0,
    ]
}

fn storage(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    // Never bound where the device has no storage buffers.
    let storage =
        if crate::downlevel::limited(device) { wgpu::BufferUsages::empty() } else { wgpu::BufferUsages::STORAGE };
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage: storage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The buffers group 2 binds, grown as frames need, and a group without
/// lights for what another camera sees (insets).
pub(crate) struct Clusters {
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    lights: wgpu::Buffer,
    ranges: wgpu::Buffer,
    indices: wgpu::Buffer,
    group: wgpu::BindGroup,
    empty: wgpu::BindGroup,
    data: Vec<f32>,
}

impl Clusters {
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let uniform = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: GRID_BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                // A grid of no lights is zeros.
                mapped_at_creation: false,
            })
        };
        let (lights, ranges, indices) = (
            storage(device, "lights", 4 * LIGHT_FLOATS as u64),
            storage(device, "light clusters", CLUSTERS as u64 * 8),
            storage(device, "light indices", 4),
        );
        let empty_uniform = uniform("no lights");
        let empty = Self::bind(device, layout, &empty_uniform, &lights, &ranges, &indices);
        let uniform = uniform("light grid");
        let group = Self::bind(device, layout, &uniform, &lights, &ranges, &indices);
        Self { layout: layout.clone(), uniform, lights, ranges, indices, group, empty, data: Vec::new() }
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        lights: &wgpu::Buffer,
        ranges: &wgpu::Buffer,
        indices: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: lights.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ranges.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: indices.as_entire_binding() },
        ];
        // The grid alone where the layout has no storage buffers.
        let bound = if crate::downlevel::limited(device) { 1 } else { entries.len() };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lights"),
            layout,
            entries: &entries[..bound],
        })
    }

    /// Sorts `frame`'s lights into the clusters of its view, on a picture
    /// of `size` pixels.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame: &RenderFrame, size: (u32, u32)) {
        let count = u32::try_from(frame.lights.len()).unwrap_or(u32::MAX);
        if count == 0 || crate::downlevel::limited(device) {
            queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&[0u32; 12]));
            return;
        }
        let assignment = assign(&frame.lights, &frame.view, &frame.projection);
        self.data.clear();
        for light in &frame.lights {
            let (p, c, d) = (&light.position, &light.color, &light.direction);
            self.data.extend([p.x, p.y, p.z, light.radius, c.x, c.y, c.z, light.outer, d.x, d.y, d.z, light.inner]);
        }
        let light_bytes: &[u8] = bytemuck::cast_slice(&self.data);
        let index_bytes: &[u8] = bytemuck::cast_slice(&assignment.indices);
        let mut grown = false;
        for (buffer, label, bytes) in
            [(&mut self.lights, "lights", light_bytes.len()), (&mut self.indices, "light indices", index_bytes.len())]
        {
            let bytes = bytes as u64;
            if bytes > buffer.size() {
                *buffer = storage(device, label, bytes.next_power_of_two());
                grown = true;
            }
        }
        if grown {
            self.group = Self::bind(device, &self.layout, &self.uniform, &self.lights, &self.ranges, &self.indices);
        }
        queue.write_buffer(&self.lights, 0, light_bytes);
        if !index_bytes.is_empty() {
            queue.write_buffer(&self.indices, 0, index_bytes);
        }
        queue.write_buffer(&self.ranges, 0, bytemuck::cast_slice(&assignment.ranges));
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&grid(count, &assignment.slicing, size)));
    }

    /// The frame's lights.
    pub fn group(&self) -> &wgpu::BindGroup {
        &self.group
    }

    /// No lights, for what another camera sees.
    pub fn empty(&self) -> &wgpu::BindGroup {
        &self.empty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::IDENTITY;
    use pen_protocol::Vector;

    fn perspective(near: f32, far: f32) -> Matrix {
        let mut m = [0.0; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = far / (near - far);
        m[11] = -1.0;
        m[14] = near * far / (near - far);
        m
    }

    fn point(x: f32, y: f32, z: f32, radius: f32) -> Light {
        Light {
            position: Vector { x, y, z },
            radius,
            color: Vector { x: 1.0, y: 1.0, z: 1.0 },
            direction: Vector::default(),
            inner: -1.0,
            outer: -1.0,
        }
    }

    fn clusters_with(assignment: &Assignment, light: u32) -> Vec<usize> {
        (0..CLUSTERS)
            .filter(|&cluster| {
                let [first, count] = assignment.ranges[cluster];
                assignment.indices[first as usize..(first + count) as usize].contains(&light)
            })
            .collect()
    }

    #[test]
    fn slices_the_depth_the_projection_sees() {
        let slicing = slicing(&perspective(0.1, 500.0), 1.0);
        // As near and far as f32 matrices say.
        assert!((slicing.near - 0.1).abs() < 1e-6 && (slicing.far - 500.0).abs() < 0.1, "{slicing:?}");
        assert_eq!(slicing.slice(0.1), 0);
        assert_eq!(slicing.slice(499.0), SLICES - 1);
        // Each slice is as many times deeper than the last.
        let middle = (0.1f64 * 500.0).sqrt();
        assert_eq!(slicing.slice(middle * 1.01), SLICES / 2);
        // An orthographic camera's slices are even; WebGPU's depth runs 0..1.
        let mut orthographic = IDENTITY;
        orthographic[10] = -0.01;
        orthographic[14] = 0.0;
        let even = super::slicing(&orthographic, 1.0);
        assert!(even.linear && even.near.abs() < 1e-3 + 1e-4 && (even.far - 100.0).abs() < 1e-3, "{even:?}");
        assert_eq!(even.slice(50.5), SLICES / 2);
    }

    #[test]
    fn a_light_lands_in_the_clusters_round_it() {
        let projection = perspective(0.1, 100.0);
        // Straight ahead, 10 deep, reaching 1: the middle tiles, a few slices.
        let lights = [point(0.0, 0.0, -10.0, 1.0), point(0.0, 0.0, 5.0, 1.0), point(-30.0, 0.0, -10.0, 1.0)];
        let assignment = assign(&lights, &IDENTITY, &projection);
        let near = clusters_with(&assignment, 0);
        assert!(!near.is_empty());
        let tiles = |cluster: usize| (cluster % 16, (cluster / 16) % 9, cluster / (16 * 9));
        let (x, y, z): (Vec<_>, Vec<_>, Vec<_>) = near.iter().map(|&cluster| tiles(cluster)).fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut xs, mut ys, mut zs), (x, y, z)| {
                xs.push(x);
                ys.push(y);
                zs.push(z);
                (xs, ys, zs)
            },
        );
        assert!(x.iter().all(|&x| (7..=8).contains(&x)), "{x:?}");
        assert!(y.iter().all(|&y| (4..=5).contains(&y)), "{y:?}");
        let slicing = assignment.slicing;
        let expected = slicing.slice(9.0)..=slicing.slice(11.0);
        assert!(z.iter().all(|z| expected.contains(&u32::try_from(*z).unwrap())), "{z:?} {expected:?}");
        // Behind the eye, and off to the side, it lights nothing.
        assert!(clusters_with(&assignment, 1).is_empty());
        assert!(clusters_with(&assignment, 2).is_empty());
    }

    #[test]
    fn every_lit_point_finds_its_light_in_its_cluster() {
        // Brute force: points ahead of the camera, each light that reaches
        // one must be in the point's cluster.
        let projection = perspective(0.1, 60.0);
        let mut view = IDENTITY;
        view[13] = -2.0;
        let mut lights = Vec::new();
        let mut seed = 7u32;
        let mut random = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f64::from(seed >> 8) / f64::from(1u32 << 24)
        };
        for index in 0..200 {
            #[allow(clippy::cast_possible_truncation, reason = "test coordinates")]
            let mut light = point(
                (random() * 40.0 - 20.0) as f32,
                (random() * 10.0 - 3.0) as f32,
                (-random() * 50.0) as f32,
                (0.5 + random() * 4.0) as f32,
            );
            if index % 3 == 0 {
                light.direction = Vector { x: 0.0, y: -1.0, z: 0.0 };
                light.outer = if index % 2 == 0 { 0.9 } else { 0.3 };
                light.inner = light.outer + 0.05;
            }
            lights.push(light);
        }
        let assignment = assign(&lights, &view, &projection);
        let grid = grid(200, &assignment.slicing, (320, 180));
        let scale = (f64::from(f32::from_bits(grid[4])), f64::from(f32::from_bits(grid[5])));
        for _ in 0..4000 {
            let (sx, sy, depth) = (random() * 320.0, random() * 180.0, 0.2 + random() * 55.0);
            // The view-space point at that pixel and depth.
            let ndc = (sx / 160.0 - 1.0, 1.0 - sy / 90.0);
            let seen = [ndc.0 * depth, ndc.1 * depth, -depth];
            let world = [seen[0], seen[1] + 2.0, seen[2]];
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "test tiles")]
            let tile = ((sx * scale.0) as u32, (sy * scale.1) as u32);
            let cluster = ((assignment.slicing.slice(depth) * TILES.1 + tile.1) * TILES.0 + tile.0) as usize;
            let [first, count] = assignment.ranges[cluster];
            let found = &assignment.indices[first as usize..(first + count) as usize];
            for (index, light) in (0u32..).zip(&lights) {
                let to = [
                    f64::from(light.position.x) - world[0],
                    f64::from(light.position.y) - world[1],
                    f64::from(light.position.z) - world[2],
                ];
                let distance = to.iter().map(|value| value * value).sum::<f64>().sqrt();
                let inside_cone = light.outer <= -1.0 || to[1] / distance >= f64::from(light.outer);
                if distance < f64::from(light.radius) && inside_cone {
                    assert!(found.contains(&index), "light {index} misses the point at {world:?}");
                }
            }
        }
    }

    #[test]
    #[ignore = "a measurement; run with --ignored --nocapture in release"]
    fn clustering_time() {
        let projection = perspective(0.1, 200.0);
        for count in [64u16, 512, 4096] {
            let lights: Vec<Light> = (0..count)
                .map(|index| {
                    let at = f32::from(index);
                    point((at * 0.37).sin() * 40.0, 1.0, -5.0 - (at * 0.11).cos().abs() * 80.0, 2.0)
                })
                .collect();
            let started = std::time::Instant::now();
            let assignment = assign(&lights, &IDENTITY, &projection);
            eprintln!(
                "{count} lights: {:.2} ms, {} indices",
                started.elapsed().as_secs_f64() * 1000.0,
                assignment.indices.len()
            );
        }
    }

    #[test]
    fn refuses_lights_without_reach() {
        assert!(validate(&[point(0.0, 0.0, 0.0, 1.0)]).is_ok());
        assert!(validate(&[point(0.0, 0.0, 0.0, 0.0)]).is_err());
        let mut cone = point(0.0, 0.0, 0.0, 1.0);
        cone.outer = 2.0;
        assert!(validate(&[cone]).is_err());
    }
}
