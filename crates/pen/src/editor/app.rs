//! Live `winit` desktop window loop for `pen edit [scene.pen-scene.json]`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use pen_render::{RenderTarget, Renderer, RendererOptions};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::editor::EditorState;
use crate::editor::gizmo::{GizmoMode, GizmoSpace};
use crate::editor::hud::{EditorHud, HudClickZones};

struct SurfaceFrame {
    size: (u32, u32),
    surface_tex: Option<wgpu::SurfaceTexture>,
    view: Option<wgpu::TextureView>,
}

impl RenderTarget for SurfaceFrame {
    fn size(&self) -> (u32, u32) {
        self.size
    }
    fn view(&mut self) -> Option<wgpu::TextureView> {
        self.view.clone()
    }
    fn present(&mut self) {}
}

struct EditorApp {
    state: EditorState,
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    renderer: Option<Renderer>,
    hud: Option<EditorHud>,
    click_zones: HudClickZones,
    cursor: (f32, f32),
    lmb: bool,
    rmb: bool,
    mmb: bool,
    modifiers: ModifiersState,
    last_frame: Instant,
    error: Option<anyhow::Error>,
}

impl EditorApp {
    fn new(state: EditorState) -> Self {
        Self {
            state,
            window: None,
            surface: None,
            config: None,
            device: None,
            queue: None,
            renderer: None,
            hud: None,
            click_zones: HudClickZones { hierarchy_rows: Vec::new(), timeline_bar: [0.0; 4] },
            cursor: (0.0, 0.0),
            lmb: false,
            rmb: false,
            mmb: false,
            modifiers: ModifiersState::empty(),
            last_frame: Instant::now(),
            error: None,
        }
    }

    fn init_gpu(&mut self, display: Box<dyn wgpu::wgt::WgpuHasDisplayHandle>, window: Arc<Window>) -> Result<()> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(display));
        let surface = instance.create_surface(window.clone())?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .context("GPU adapter required for pen edit")?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: pen_render::optional_features(&adapter),
            ..Default::default()
        }))?;
        let caps = surface.get_capabilities(&adapter);
        let (format, shown) = pen_render::choose_surface(&caps, &surface.display_hdr_info(&adapter), false);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: shown.color_space,
        };
        surface.configure(&device, &config);

        let options = RendererOptions {
            format,
            display: shown,
            assets: self.state.assets_dir.clone(),
            veil: None,
            pipeline_cache: None,
        };
        let mut renderer = Renderer::new(device.clone(), queue.clone(), options)?;
        renderer.preload(&self.state.catalog);
        let hud = EditorHud::new(&device, &queue, format);

        self.window = Some(window);
        self.surface = Some(surface);
        self.config = Some(config);
        self.device = Some(device);
        self.queue = Some(queue);
        self.renderer = Some(renderer);
        self.hud = Some(hud);
        Ok(())
    }

    fn draw(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        self.state.tick(dt);

        let (Some(surface), Some(config), Some(device), Some(queue), Some(renderer), Some(hud)) = (
            self.surface.as_ref(),
            self.config.as_ref(),
            self.device.as_ref(),
            self.queue.as_ref(),
            self.renderer.as_mut(),
            self.hud.as_mut(),
        ) else {
            return;
        };
        let surface_tex = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(tex) | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                surface.configure(device, config);
                return;
            }
            _ => return,
        };
        let view = surface_tex.texture.create_view(&Default::default());
        let size = (config.width, config.height);
        let frame = self.state.build_render_frame(size);
        let mut target = SurfaceFrame { size, surface_tex: Some(surface_tex), view: Some(view.clone()) };
        let _ = renderer.submit(&frame, &self.state.catalog, &mut target);

        let mut enc = device.create_command_encoder(&Default::default());
        self.click_zones = hud.render(device, queue, &mut enc, &view, size, &self.state);
        queue.submit(Some(enc.finish()));
        if let Some(st) = target.surface_tex.take() {
            queue.present(st);
        }
    }
}

impl ApplicationHandler for EditorApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(format!("Pen Scene Editor — {}", self.state.scene.name))
            .with_inner_size(LogicalSize::new(1440.0, 900.0));
        match event_loop.create_window(attrs).map(Arc::new) {
            Ok(window) => {
                let display = Box::new(event_loop.owned_display_handle());
                if let Err(e) = self.init_gpu(display, window) {
                    self.error = Some(e);
                    event_loop.exit();
                }
            }
            Err(e) => {
                self.error = Some(e.into());
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if let (Some(surface), Some(config), Some(device)) =
                    (self.surface.as_ref(), self.config.as_mut(), self.device.as_ref())
                    && new_size.width > 0
                    && new_size.height > 0
                {
                    config.width = new_size.width;
                    config.height = new_size.height;
                    surface.configure(device, config);
                }
            }
            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),
            WindowEvent::CursorMoved { position, .. } => {
                let nx = position.x as f32;
                let ny = position.y as f32;
                let dx = nx - self.cursor.0;
                let dy = ny - self.cursor.1;
                self.cursor = (nx, ny);
                let size = self.config.as_ref().map_or((1440, 900), |c| (c.width, c.height));
                if self.rmb {
                    self.state.camera.orbit(dx, dy);
                } else if self.mmb {
                    self.state.camera.pan(dx, dy);
                } else if self.lmb && self.state.active_drag.is_some() {
                    self.state.update_gizmo_drag(nx, ny, size, self.modifiers.control_key());
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y as f32) * 0.02,
                };
                self.state.camera.zoom(scroll);
            }
            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                let pressed = btn_state == ElementState::Pressed;
                let size = self.config.as_ref().map_or((1440, 900), |c| (c.width, c.height));
                match button {
                    MouseButton::Right => self.rmb = pressed,
                    MouseButton::Middle => self.mmb = pressed,
                    MouseButton::Left => {
                        self.lmb = pressed;
                        if pressed {
                            let (x, y) = self.cursor;
                            // Check hierarchy tree click first.
                            if let Some((_, _, _, _, node_id)) = self
                                .click_zones
                                .hierarchy_rows
                                .iter()
                                .find(|&&(rx, ry, rw, rh, _)| x >= rx && x <= rx + rw && y >= ry && y <= ry + rh)
                            {
                                self.state.selected = Some(*node_id);
                                return;
                            }
                            // Check timeline scrubber click.
                            let [tx, ty, tw, th] = self.click_zones.timeline_bar;
                            if x >= tx && x <= tx + tw && y >= ty - 4.0 && y <= ty + th + 4.0 && tw > 1.0 {
                                let frac = ((x - tx) / tw).clamp(0.0, 1.0);
                                let t = frac * self.state.sequence.duration;
                                self.state.scrub_timeline(t);
                                return;
                            }
                            // Otherwise attempt gizmo drag start or 3D entity picking.
                            if !self.state.begin_gizmo_drag(x, y, size) {
                                self.state.pick_at(x, y, size);
                            }
                        } else {
                            self.state.end_gizmo_drag();
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let ctrl = self.modifiers.control_key();
                    match code {
                        KeyCode::KeyW if !self.rmb => self.state.gizmo_mode = GizmoMode::Translate,
                        KeyCode::KeyE => self.state.gizmo_mode = GizmoMode::Rotate,
                        KeyCode::KeyR => self.state.gizmo_mode = GizmoMode::Scale,
                        KeyCode::KeyG => {
                            self.state.gizmo_space = match self.state.gizmo_space {
                                GizmoSpace::World => GizmoSpace::Local,
                                GizmoSpace::Local => GizmoSpace::World,
                            };
                        }
                        KeyCode::KeyF => self.state.focus_selected(),
                        KeyCode::KeyP => self.state.toggle_play(),
                        KeyCode::KeyT => self.state.cycle_tonemap(),
                        KeyCode::KeyK => self.state.toggle_physical_sky(),
                        KeyCode::KeyD if !ctrl => self.state.toggle_ddgi(),
                        KeyCode::KeyH => self.state.toggle_selected_visible(),
                        KeyCode::BracketLeft => self.state.adjust_selected_light_intensity(-0.25),
                        KeyCode::BracketRight => self.state.adjust_selected_light_intensity(0.25),
                        KeyCode::KeyS if ctrl => {
                            let _ = self.state.save();
                        }
                        KeyCode::KeyZ if ctrl && self.modifiers.shift_key() => {
                            self.state.redo();
                        }
                        KeyCode::KeyZ if ctrl => {
                            self.state.undo();
                        }
                        KeyCode::KeyY if ctrl => {
                            self.state.redo();
                        }
                        KeyCode::Delete | KeyCode::Backspace => {
                            self.state.delete_selected();
                        }
                        KeyCode::KeyN if ctrl => {
                            self.state.add_cube_node();
                        }
                        KeyCode::KeyW if self.rmb => self.state.camera.fly(0.5, 0.0, 0.0),
                        KeyCode::KeyS if self.rmb => self.state.camera.fly(-0.5, 0.0, 0.0),
                        KeyCode::KeyA if self.rmb => self.state.camera.fly(0.0, -0.5, 0.0),
                        KeyCode::KeyD if self.rmb => self.state.camera.fly(0.0, 0.5, 0.0),
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.draw();
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

pub fn run(scene_path: Option<PathBuf>, assets_dir: PathBuf) -> Result<()> {
    let state = EditorState::open_or_create(scene_path, assets_dir)?;
    let event_loop = EventLoop::new()?;
    let mut app = EditorApp::new(state);
    event_loop.run_app(&mut app)?;
    if let Some(err) = app.error {
        return Err(err);
    }
    Ok(())
}
