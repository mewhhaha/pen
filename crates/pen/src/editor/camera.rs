//! Viewport Orbit / Pan / Zoom / Fly camera and screen-to-world ray unprojection.

use pen_scene::{look_at, normalize3, perspective};

/// Interactive 3D editor viewport camera (orbit target + spherical yaw/pitch/distance + fly step).
#[derive(Clone, Copy, Debug)]
pub struct EditorCamera {
    pub target: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub fov_deg: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for EditorCamera {
    fn default() -> Self {
        Self {
            target: [0.0, 0.5, 0.0],
            yaw: 0.65,
            pitch: 0.42,
            distance: 9.0,
            fov_deg: 55.0,
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl EditorCamera {
    pub fn eye(&self) -> [f32; 3] {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        [
            self.target[0] + self.distance * cp * sy,
            self.target[1] + self.distance * sp,
            self.target[2] + self.distance * cp * cy,
        ]
    }

    pub fn forward(&self) -> [f32; 3] {
        let eye = self.eye();
        normalize3([self.target[0] - eye[0], self.target[1] - eye[1], self.target[2] - eye[2]])
    }

    pub fn right(&self) -> [f32; 3] {
        let f = self.forward();
        // cross(forward, up(0,1,0)) = (-f.z, 0, f.x)
        normalize3([-f[2], 0.0, f[0]])
    }

    pub fn up(&self) -> [f32; 3] {
        let f = self.forward();
        let r = self.right();
        normalize3([
            r[1] * f[2] - r[2] * f[1],
            r[2] * f[0] - r[0] * f[2],
            r[0] * f[1] - r[1] * f[0],
        ])
    }

    pub fn view_matrix(&self) -> [f32; 16] {
        look_at(self.eye(), self.target, [0.0, 1.0, 0.0])
    }

    pub fn projection_matrix(&self, aspect: f32) -> [f32; 16] {
        perspective(self.fov_deg, aspect, self.near, self.far)
    }

    pub fn orbit(&mut self, dx_px: f32, dy_px: f32) {
        self.yaw -= dx_px * 0.007;
        self.pitch = (self.pitch + dy_px * 0.007).clamp(-1.48, 1.48);
    }

    pub fn pan(&mut self, dx_px: f32, dy_px: f32) {
        let scale = self.distance * 0.0018;
        let r = self.right();
        let u = self.up();
        for k in 0..3 {
            self.target[k] += (-r[k] * dx_px + u[k] * dy_px) * scale;
        }
    }

    pub fn zoom(&mut self, scroll_delta: f32) {
        let factor = (1.0 - scroll_delta * 0.12).clamp(0.5, 2.0);
        self.distance = (self.distance * factor).clamp(0.5, 400.0);
    }

    pub fn fly(&mut self, forward_step: f32, right_step: f32, up_step: f32) {
        let f = self.forward();
        let r = self.right();
        for k in 0..3 {
            self.target[k] += f[k] * forward_step + r[k] * right_step;
        }
        self.target[1] += up_step;
    }

    pub fn focus_on(&mut self, center: [f32; 3], radius: f32) {
        self.target = center;
        self.distance = (radius * 2.8).clamp(2.0, 80.0);
    }

    /// Unprojects a screen pixel `(px, py)` in a `(width, height)` viewport into a world ray `(origin, dir)`.
    pub fn screen_ray(&self, px: f32, py: f32, width: f32, height: f32) -> ([f32; 3], [f32; 3]) {
        let aspect = (width / height.max(1.0)).max(0.05);
        let ndc_x = (px / width.max(1.0)) * 2.0 - 1.0;
        let ndc_y = 1.0 - (py / height.max(1.0)) * 2.0;
        let half_h = (self.fov_deg.to_radians() * 0.5).tan();
        let half_w = half_h * aspect;
        let f = self.forward();
        let r = self.right();
        let u = self.up();
        let dir = normalize3([
            f[0] + r[0] * (ndc_x * half_w) + u[0] * (ndc_y * half_h),
            f[1] + r[1] * (ndc_x * half_w) + u[1] * (ndc_y * half_h),
            f[2] + r[2] * (ndc_x * half_w) + u[2] * (ndc_y * half_h),
        ]);
        (self.eye(), dir)
    }

    /// Projects a world-space point into screen pixels `(px, py)` if in front of the camera.
    pub fn project_point(&self, world: [f32; 3], width: f32, height: f32) -> Option<[f32; 2]> {
        let aspect = (width / height.max(1.0)).max(0.05);
        let v = self.view_matrix();
        let p = self.projection_matrix(aspect);
        let vx = v[0] * world[0] + v[4] * world[1] + v[8] * world[2] + v[12];
        let vy = v[1] * world[0] + v[5] * world[1] + v[9] * world[2] + v[13];
        let vz = v[2] * world[0] + v[6] * world[1] + v[10] * world[2] + v[14];
        let vw = v[3] * world[0] + v[7] * world[1] + v[11] * world[2] + v[15];
        let cx = p[0] * vx + p[4] * vy + p[8] * vz + p[12] * vw;
        let cy = p[1] * vx + p[5] * vy + p[9] * vz + p[13] * vw;
        let cw = p[3] * vx + p[7] * vy + p[11] * vz + p[15] * vw;
        if cw <= 0.05 {
            return None;
        }
        let ndc_x = cx / cw;
        let ndc_y = cy / cw;
        Some([(ndc_x * 0.5 + 0.5) * width, (0.5 - ndc_y * 0.5) * height])
    }
}
