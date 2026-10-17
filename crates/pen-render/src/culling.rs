// What a frame need not draw: each mesh's bounding sphere, worked out when it
// loads, placed by a draw's transform and tested against the clip volume of
// each view the frame draws (the camera's, and the sun's shadow maps'). A
// draw whose sphere lies wholly outside a view's volume would put no pixel
// there, so leaving it out changes nothing drawn.
//
// Only a shader that declares how far its vertex stage may move a vertex from
// where its mesh puts it (`const bounds_margin = ...;`) is culled: without it
// the renderer cannot know where its vertices land.
use pen_protocol::Matrix;

use crate::matrix::{multiply, narrow};

/// The most views a frame tests its draws against: the camera's and up to
/// four shadow maps'.
pub(crate) const MAX_VIEWS: usize = 5;

/// A draw seen by every view, or not culled.
pub(crate) const ALL_VIEWS: u8 = (1 << MAX_VIEWS) - 1;

/// The camera's view, the first.
pub(crate) const CAMERA: u8 = 1;

/// A sphere holding every vertex of a mesh, in its own space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
}

impl Sphere {
    /// The sphere round the positions that start every `stride` floats of
    /// `vertices`: centred on their box, as far out as the farthest.
    pub fn around(vertices: &[f32], stride: usize) -> Sphere {
        let mut low = [f64::INFINITY; 3];
        let mut high = [f64::NEG_INFINITY; 3];
        for vertex in vertices.chunks_exact(stride) {
            for axis in 0..3 {
                low[axis] = low[axis].min(f64::from(vertex[axis]));
                high[axis] = high[axis].max(f64::from(vertex[axis]));
            }
        }
        if low[0] > high[0] {
            return Sphere::default();
        }
        let center = [0, 1, 2].map(|axis| (low[axis] + high[axis]) / 2.0);
        let farthest = vertices
            .chunks_exact(stride)
            .map(|vertex| (0..3).map(|axis| (f64::from(vertex[axis]) - center[axis]).powi(2)).sum::<f64>())
            .fold(0.0, f64::max);
        // Rounded outward, so that no vertex lies outside it in f32.
        Sphere { center: center.map(narrow), radius: narrow(farthest.sqrt() * (1.0 + 1e-5) + 1e-6) }
    }

    /// Grown by `margin` and placed by `transform`, its radius stretched as
    /// the transform stretches anything (`stretch`).
    pub fn placed(&self, transform: &Matrix, margin: f32) -> Sphere {
        let m = transform;
        let [x, y, z] = self.center;
        let center = [0, 1, 2].map(|row| m[row] * x + m[4 + row] * y + m[8 + row] * z + m[12 + row]);
        Sphere { center, radius: (self.radius + margin.max(0.0)) * stretch(m) }
    }
}

/// The most a transform stretches any length: its longest axis when its
/// axes are square to each other (a rotation and a scale along each),
/// otherwise (a shear) the root of their squared lengths' sum, which bounds
/// the stretch whatever the shear. The culling shader (cull.wgsl) works it
/// out alike.
pub(crate) fn stretch(m: &Matrix) -> f32 {
    let dot = |a: usize, b: usize| m[a * 4] * m[b * 4] + m[a * 4 + 1] * m[b * 4 + 1] + m[a * 4 + 2] * m[b * 4 + 2];
    let squared = [dot(0, 0), dot(1, 1), dot(2, 2)];
    // Square within float error; compared squared, to spare the roots.
    let square = [(0, 1), (0, 2), (1, 2)].iter().all(|&(a, b)| dot(a, b) * dot(a, b) <= 1e-8 * squared[a] * squared[b]);
    if square {
        squared[0].max(squared[1]).max(squared[2]).sqrt()
    } else {
        (squared[0] + squared[1] + squared[2]).sqrt()
    }
}

/// What picks a draw's level of detail: the row of the camera's
/// view-projection that gives a point's w (its distance ahead, in a
/// perspective), and how many of the scene's pixels a unit spans at w = 1.
/// With no pixels, every mesh draws whole.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Detail {
    pub row: [f32; 4],
    pub pixels: f32,
}

impl Detail {
    /// For a camera of `projection` and `view` over a scene `height` pixels high.
    pub fn new(projection: &Matrix, view: &Matrix, height: f32) -> Detail {
        let m = multiply(projection, view);
        Detail { row: [m[3], m[7], m[11], m[15]], pixels: projection[5].abs() * height / 2.0 }
    }

    /// The simplest of a mesh's levels (their errors in its units, the whole
    /// mesh's first) whose error, on `placed` stretched by `stretch`, spans
    /// at most a pixel where its nearest point is. The culling shader picks
    /// alike.
    pub fn level(&self, placed: &Sphere, stretch: f32, errors: &[f32]) -> usize {
        if self.pixels <= 0.0 || errors.len() < 2 {
            return 0;
        }
        let [a, b, c, d] = self.row;
        let [x, y, z] = placed.center;
        let reach = (a * a + b * b + c * c).sqrt();
        let nearest = (a * x + b * y + c * z + d - placed.radius * reach).max(1e-4);
        let per_unit = self.pixels * stretch / nearest;
        (1..errors.len()).rev().find(|&level| errors[level] * per_unit <= 1.0).unwrap_or(0)
    }
}

/// The planes bounding a view's clip volume (WebGPU's: x and y within -w..w,
/// z within 0..w), each `[a, b, c, d]` with its normal of unit length
/// pointing in, so that a point inside has `a x + b y + c z + d >= 0`. A
/// plane a projection lacks (an infinitely far one) is all zeros, which
/// every point is inside.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Frustum {
    planes: [[f32; 4]; 6],
}

impl Frustum {
    /// The volume `projection * view` sees.
    pub fn of(projection: &Matrix, view: &Matrix) -> Frustum {
        Frustum::new(&multiply(projection, view))
    }

    /// The volume a column-major view-projection matrix sees (Gribb and
    /// Hartmann's planes from its rows).
    pub fn new(m: &Matrix) -> Frustum {
        let row = |index: usize| [m[index], m[4 + index], m[8 + index], m[12 + index]].map(f64::from);
        let (x, y, z, w) = (row(0), row(1), row(2), row(3));
        let sum = |a: [f64; 4], b: [f64; 4], sign: f64| [0, 1, 2, 3].map(|lane| a[lane] + sign * b[lane]);
        let planes = [sum(w, x, 1.0), sum(w, x, -1.0), sum(w, y, 1.0), sum(w, y, -1.0), z, sum(w, z, -1.0)];
        Frustum {
            planes: planes.map(|plane| {
                let length = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]).sqrt();
                if length > 1e-12 && length.is_finite() { plane.map(|lane| narrow(lane / length)) } else { [0.0; 4] }
            }),
        }
    }

    /// Whether any of the sphere may lie inside.
    pub fn sees(&self, sphere: &Sphere) -> bool {
        let [x, y, z] = sphere.center;
        // A hair of slack against rounding: culling must never take away
        // what would have been drawn.
        let slack = sphere.radius * 1e-4 + 1e-5;
        self.planes.iter().all(|[a, b, c, d]| a * x + b * y + c * z + d >= -(sphere.radius + slack))
    }
}

/// The views a frame draws, the camera's first.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Views {
    frustums: [Frustum; MAX_VIEWS],
    count: usize,
}

impl Views {
    pub fn new(camera: Frustum) -> Views {
        let mut views = Views::default();
        views.frustums[0] = camera;
        views.count = 1;
        views
    }

    /// Adds a view; it is view `count()` before.
    pub fn push(&mut self, frustum: Frustum) {
        assert!(self.count < MAX_VIEWS, "at most MAX_VIEWS views");
        self.frustums[self.count] = frustum;
        self.count += 1;
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// Each view's six planes, in order.
    pub fn planes(&self) -> impl Iterator<Item = &[f32; 4]> {
        self.frustums[..self.count].iter().flat_map(|frustum| &frustum.planes)
    }

    /// Which views (a bit each, the camera's first) may see the sphere; a
    /// view past `count()` sees nothing.
    pub fn mask(&self, sphere: &Sphere) -> u8 {
        let mut seen = 0;
        for (index, frustum) in self.frustums[..self.count].iter().enumerate() {
            if frustum.sees(sphere) {
                seen |= 1 << index;
            }
        }
        seen
    }

    /// Every view there is.
    pub fn all(&self) -> u8 {
        u8::try_from((1u32 << self.count) - 1).expect("MAX_VIEWS fits a byte")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::IDENTITY;

    fn translated(x: f32, y: f32, z: f32) -> Matrix {
        let mut m = IDENTITY;
        m[12] = x;
        m[13] = y;
        m[14] = z;
        m
    }

    /// A camera at the origin looking down -z, 90 degrees up and down, from
    /// 0.1 to 100 (WebGPU's 0..1 depth).
    fn perspective() -> Matrix {
        let (near, far) = (0.1f32, 100.0f32);
        let mut m = [0.0; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = far / (near - far);
        m[11] = -1.0;
        m[14] = near * far / (near - far);
        m
    }

    #[test]
    fn spheres_hold_their_vertices_wherever_they_are_placed() {
        // Three vertices of a stride-4 array (the fourth float is not a
        // coordinate).
        let vertices = [0.0, 0.0, 0.0, 9.0, 2.0, 0.0, 0.0, 9.0, 0.0, 4.0, 0.0, 9.0];
        let sphere = Sphere::around(&vertices, 4);
        assert_eq!(sphere.center, [1.0, 2.0, 0.0]);
        assert!(sphere.radius >= 5f32.sqrt() && sphere.radius < 5f32.sqrt() + 1e-3);
        assert_eq!(Sphere::around(&[], 4), Sphere::default());
        // Scaled 3 along x and moved: the radius takes the longest axis.
        let mut stretch = translated(10.0, 0.0, 0.0);
        stretch[0] = 3.0;
        let placed = sphere.placed(&stretch, 0.5);
        assert_eq!(placed.center, [13.0, 2.0, 0.0]);
        assert!((placed.radius - (sphere.radius + 0.5) * 3.0).abs() < 1e-5);
        // A shear: every axis counts.
        let mut shear = IDENTITY;
        shear[4] = 1.0;
        assert!((sphere.placed(&shear, 0.0).radius - sphere.radius * 2.0).abs() < 1e-4);
    }

    #[test]
    fn frustums_see_what_their_clip_volume_holds() {
        let camera = Frustum::new(&perspective());
        let ball = |x: f32, y: f32, z: f32, radius: f32| Sphere { center: [x, y, z], radius };
        assert!(camera.sees(&ball(0.0, 0.0, -10.0, 1.0)));
        // Behind the eye, beyond the far plane, off to each side.
        assert!(!camera.sees(&ball(0.0, 0.0, 5.0, 1.0)));
        assert!(!camera.sees(&ball(0.0, 0.0, -102.0, 1.0)));
        assert!(!camera.sees(&ball(12.0, 0.0, -10.0, 1.0)));
        assert!(!camera.sees(&ball(0.0, -12.0, -10.0, 1.0)));
        // Touching the edge is seen: x = z is the right plane at 45 degrees.
        assert!(camera.sees(&ball(11.0, 0.0, -10.0, 0.75)));
        // The identity's volume is the clip box itself.
        let clip = Frustum::new(&IDENTITY);
        assert!(clip.sees(&ball(0.0, 0.0, 0.5, 0.1)));
        assert!(!clip.sees(&ball(2.0, 0.0, 0.5, 0.5)));
        assert!(!clip.sees(&ball(0.0, 0.0, -1.0, 0.5)));
        // A projection without a far plane (infinite reversed depth).
        let mut infinite = perspective();
        infinite[10] = 0.0;
        infinite[14] = 0.1;
        assert!(Frustum::new(&infinite).sees(&ball(0.0, 0.0, -1e6, 1.0)));
        assert!(!Frustum::new(&infinite).sees(&ball(0.0, 0.0, 5.0, 1.0)));
    }

    #[test]
    fn levels_of_detail_follow_distance_and_scale() {
        // 640 pixels high: a unit at w = 1 spans 320 of them.
        let detail = Detail::new(&perspective(), &IDENTITY, 640.0);
        assert_eq!(detail.pixels, 320.0);
        let errors = [0.0, 0.01, 0.1];
        let at = |z: f32| Sphere { center: [0.0, 0.0, z], radius: 1.0 };
        // Nearest at w = 2: a hundredth spans 1.6 pixels, so whole.
        assert_eq!(detail.level(&at(-3.0), 1.0, &errors), 0);
        // At w = 9: 0.36 pixels for the first level, 3.6 for the second.
        assert_eq!(detail.level(&at(-10.0), 1.0, &errors), 1);
        // Three times the size, three times the error: 1.07 pixels.
        assert_eq!(detail.level(&at(-10.0), 3.0, &errors), 0);
        assert_eq!(detail.level(&at(-100.0), 1.0, &errors), 2);
        // The eye inside it, or no levels, or no pixels: whole.
        assert_eq!(detail.level(&at(0.0), 1.0, &errors), 0);
        assert_eq!(detail.level(&at(-100.0), 1.0, &errors[..1]), 0);
        assert_eq!(Detail::default().level(&at(-100.0), 1.0, &errors), 0);
    }

    #[test]
    fn views_mark_which_of_them_see_a_sphere() {
        let mut views = Views::new(Frustum::new(&perspective()));
        views.push(Frustum::new(&IDENTITY));
        assert_eq!((views.count(), views.all()), (2, 0b11));
        let near_origin = Sphere { center: [0.0, 0.0, 0.5], radius: 0.2 };
        // Behind the camera's near plane, inside the clip box.
        assert_eq!(views.mask(&near_origin), 0b10);
        assert_eq!(views.mask(&Sphere { center: [0.0, 0.0, -10.0], radius: 1.0 }), 0b01);
        assert_eq!(views.mask(&Sphere { center: [0.0, 0.0, -0.2], radius: 1.0 }), 0b11);
        assert_eq!(ALL_VIEWS, 0b11111);
    }
}
