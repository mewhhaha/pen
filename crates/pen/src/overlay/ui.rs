//! The overlay itself: egui over the window, its input routing, and the
//! panels' layout and actions (see the module's documentation).
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use egui::{Color32, RichText};
use pen_protocol::GameEvent;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

use super::commands::{Action, LEVELS, Table};
use super::log;
use super::panels::{self, Console, Details, Profiler, Reload, TimelinePanel, Travel, Tunables};
use crate::session::{Gpu, Overlay, Session};

/// How the overlay starts.
pub struct Settings {
    /// The checkout: its command table, and where recordings and shots go.
    pub project: Option<PathBuf>,
    /// Shown from the start (`--overlay`, PEN_OVERLAY=1).
    pub open: bool,
}

/// Which panels show while the overlay does.
struct Shown {
    console: bool,
    profiler: bool,
    timeline: bool,
    reload: bool,
    tunables: bool,
}

/// egui's triangles for the last layout, drawn onto each image shown until
/// the next.
struct Painter {
    renderer: egui_wgpu::Renderer,
    primitives: Vec<egui::ClippedPrimitive>,
    pixels_per_point: f32,
    /// Texture changes not yet made: uploads before the next paint, and
    /// frees after it.
    textures: egui::TexturesDelta,
}

impl Painter {
    fn paint(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) {
        for (id, deltas) in &self.textures.set {
            for delta in deltas {
                self.renderer.update_texture(device, queue, *id, delta);
            }
        }
        self.textures.set.clear();
        if !self.primitives.is_empty() {
            let screen = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [target.width(), target.height()],
                pixels_per_point: self.pixels_per_point,
            };
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("overlay") });
            let uploads = self.renderer.update_buffers(device, queue, &mut encoder, &self.primitives, &screen);
            let view = target.create_view(&Default::default());
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("overlay"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.renderer.render(&mut pass, &self.primitives, &screen);
            drop(pass);
            queue.submit(uploads.into_iter().chain([encoder.finish()]));
        }
        for id in self.textures.free.drain() {
            self.renderer.free_texture(&id);
        }
    }
}

impl Drop for Painter {
    fn drop(&mut self) {
        // Changes never made go with the renderer.
        self.textures.clear();
    }
}

/// The dev overlay (`session::Overlay`).
pub struct DevOverlay {
    context: egui::Context,
    winit: egui_winit::State,
    painter: Painter,
    open: bool,
    shown: Shown,
    console: Console,
    profiler: Profiler,
    timeline: TimelinePanel,
    reload: Reload,
    tunables: Tunables,
    project: Option<PathBuf>,
    adapter: String,
    /// The log's lines read so far.
    serial: u64,
    /// Frames are being timed for the profiler.
    measuring: bool,
    /// A panel has the keyboard: the game gets no keys.
    typing: bool,
    /// The pointer is over a panel.
    over: bool,
    /// Buttons the game got pressed, whose drags and releases stay its own.
    buttons: u32,
    previous: Option<Instant>,
    /// Frames each panel has shown for (`pin`).
    placed: [u32; 5],
    /// egui's AccessKit tree from the last layout, until taken
    /// (`accessibility` grafts it into the window's).
    accesskit: Option<accesskit::TreeUpdate>,
    /// Assistive technology's requests, for the next layout.
    accesskit_requests: Vec<accesskit::ActionRequest>,
}

/// A name for a new recording or shot.
fn stamp() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |since| since.as_secs())
}

/// A panel's window, held at `pos` by its `pivot` corner. A new egui window
/// is first laid out in a sizing pass that can make it far taller than it
/// is (wrapping text before its width is known), and the screen's edge would
/// push it away from its corner for good; so for its first frames it is put
/// back (`placed` counts them), and after that it moves only when dragged.
fn pin<'a>(window: egui::Window<'a>, pivot: egui::Align2, pos: [f32; 2], placed: &mut u32) -> egui::Window<'a> {
    let window = window.pivot(pivot).default_pos(pos);
    *placed += 1;
    if *placed <= 3 { window.current_pos(pos) } else { window }
}

fn button_bit(button: MouseButton) -> u32 {
    match button {
        MouseButton::Left => 1,
        MouseButton::Right => 2,
        MouseButton::Middle => 4,
        MouseButton::Back => 8,
        MouseButton::Forward => 16,
        MouseButton::Other(_) => 32,
    }
}

impl DevOverlay {
    pub fn new(window: &Window, gpu: &Gpu, developing: bool, settings: Settings) -> DevOverlay {
        let context = egui::Context::default();
        let mut visuals = egui::Visuals::dark();
        visuals.window_fill = Color32::from_rgba_unmultiplied(18, 20, 24, 232);
        visuals.panel_fill = visuals.window_fill;
        visuals.window_shadow = egui::Shadow::NONE;
        context.set_visuals(visuals);
        // Its widgets for assistive technology too, under the game's.
        context.enable_accesskit();
        let max_texture_side = gpu.device.limits().max_texture_dimension_2d as usize;
        let winit = egui_winit::State::new(
            context.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            Some(max_texture_side),
        );
        // The swapchain is not sRGB, and egui's dithering assumes it is.
        let options = egui_wgpu::RendererOptions { dithering: false, ..Default::default() };
        let renderer = egui_wgpu::Renderer::new(&gpu.device, gpu.format, options);
        let info = gpu.device.adapter_info();
        let driver = [info.driver.as_str(), info.driver_info.as_str()].join(" ");
        let adapter = format!("{} ({:?}, {:?}) {}", info.name, info.backend, info.device_type, driver.trim());
        let table = settings.project.as_deref().map(Table::load).unwrap_or_default();
        let mut overlay = DevOverlay {
            context,
            winit,
            painter: Painter {
                renderer,
                primitives: Vec::new(),
                pixels_per_point: 1.0,
                textures: egui::TexturesDelta::default(),
            },
            open: settings.open,
            shown: Shown { console: true, profiler: true, timeline: true, reload: developing, tunables: false },
            console: Console::new(table),
            profiler: Profiler::new(),
            timeline: TimelinePanel::new(PathBuf::new()),
            reload: Reload::new(developing),
            tunables: Tunables::load(settings.project.as_deref()),
            project: settings.project,
            adapter,
            serial: 0,
            measuring: false,
            typing: false,
            over: false,
            buttons: 0,
            previous: None,
            placed: [0; 5],
            accesskit: None,
            accesskit_requests: Vec::new(),
        };
        overlay.timeline.next_path(overlay.new_file("recordings", "penrec"));
        overlay.console.tunables(&overlay.tunables);
        overlay
    }

    /// A new file in the checkout's `.pen/DIRECTORY` (or `DIRECTORY` here).
    fn new_file(&self, directory: &str, extension: &str) -> PathBuf {
        let base = match &self.project {
            Some(root) => root.join(".pen").join(directory),
            None => PathBuf::from(directory),
        };
        base.join(format!("{}.{extension}", stamp()))
    }

    fn toggle(&mut self) {
        self.open = !self.open;
        if self.open {
            self.console.focus = true;
        } else {
            self.context.memory_mut(|memory| {
                if let Some(id) = memory.focused() {
                    memory.surrender_focus(id);
                }
            });
            self.typing = false;
            self.over = false;
        }
    }

    /// The log's new lines, into the console and the hot reload panel.
    fn read_log(&mut self) {
        for line in log::since(self.serial) {
            self.serial = line.serial + 1;
            self.console.take(&line);
            self.reload.take(&line);
        }
    }

    /// The panels, laid out; what they ask for is done after.
    fn layout(
        &mut self,
        context: &egui::Context,
        session: &Session,
        actions: &mut Vec<Action>,
        travel: &mut Vec<Travel>,
    ) {
        // Each panel starts in a corner of its own, held by that corner as it grows.
        let screen = context.content_rect();
        let (left, right, top, bottom) =
            (screen.left() + 8.0, screen.right() - 8.0, screen.top() + 40.0, screen.bottom() - 8.0);
        let status = session.timeline();
        self.toolbar(context, &status, session.input_recording().is_some());
        if self.shown.console {
            let width = (screen.width() * 0.55).min(540.0);
            let window = egui::Window::new("Console").open(&mut self.shown.console);
            pin(window, egui::Align2::LEFT_BOTTOM, [left, bottom], &mut self.placed[0]).resizable(false).show(
                context,
                |ui| {
                    ui.set_width(width);
                    self.console.show(ui, actions);
                },
            );
        }
        if self.shown.profiler {
            let details = Details {
                stats: session.gpu.renderer.stats(),
                memory: Some(session.guest_memory()).filter(|bytes| *bytes > 0),
                adapter: self.adapter.clone(),
                size: session.gpu.size,
                timeline: status,
            };
            let window = egui::Window::new("Profiler").open(&mut self.shown.profiler);
            pin(window, egui::Align2::RIGHT_TOP, [right, top], &mut self.placed[1]).resizable(false).show(
                context,
                |ui| {
                    ui.set_width(260.0);
                    self.profiler.show(ui, &details);
                },
            );
        }
        if self.shown.timeline {
            let recording = session.input_recording().map(|(path, from)| (path.display().to_string(), from));
            let window = egui::Window::new("Timeline").open(&mut self.shown.timeline);
            pin(window, egui::Align2::RIGHT_BOTTOM, [right, bottom], &mut self.placed[2]).resizable(false).show(
                context,
                |ui| {
                    ui.set_width(340.0);
                    travel.extend(self.timeline.show(ui, &status, recording));
                },
            );
        }
        if self.shown.reload {
            let window = egui::Window::new("Hot reload").open(&mut self.shown.reload);
            pin(window, egui::Align2::LEFT_TOP, [left, top], &mut self.placed[3]).resizable(false).show(
                context,
                |ui| {
                    ui.set_width(340.0);
                    self.reload.show(ui);
                },
            );
        }
        if self.shown.tunables {
            let mut tuned = Vec::new();
            let window = egui::Window::new("Tunables").open(&mut self.shown.tunables);
            pin(window, egui::Align2::CENTER_TOP, [screen.center().x, top], &mut self.placed[4]).resizable(false).show(
                context,
                |ui| {
                    ui.set_width(420.0);
                    self.tunables.show(ui, &mut tuned);
                },
            );
            actions.extend(tuned);
        }
    }

    /// The strip at the top: which panels show, and where the game stands.
    fn toolbar(&mut self, context: &egui::Context, status: &crate::session::TimelineStatus, recording: bool) {
        egui::Area::new(egui::Id::new("pen overlay toolbar"))
            .anchor(egui::Align2::LEFT_TOP, [8.0, 6.0])
            .order(egui::Order::Foreground)
            .show(context, |ui| {
                egui::Frame::window(ui.style()).inner_margin(egui::Margin::symmetric(8, 4)).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("pen").strong());
                        ui.toggle_value(&mut self.shown.console, "Console");
                        ui.toggle_value(&mut self.shown.profiler, "Profiler");
                        ui.toggle_value(&mut self.shown.timeline, "Timeline");
                        let reload = if self.reload.failing() {
                            RichText::new("Hot reload").color(panels::ERROR)
                        } else {
                            RichText::new("Hot reload")
                        };
                        ui.toggle_value(&mut self.shown.reload, reload);
                        ui.toggle_value(&mut self.shown.tunables, "Tunables");
                        ui.separator();
                        ui.label(RichText::new(format!("frame {}", status.frame)).monospace());
                        if status.paused {
                            ui.label(RichText::new("paused").color(Color32::from_rgb(255, 200, 90)));
                        }
                        if status.replaying {
                            ui.label(RichText::new("replaying").color(panels::COMMAND));
                        }
                        if recording {
                            ui.label(RichText::new("rec").color(panels::ERROR).strong());
                        }
                        ui.label(RichText::new("` hides").color(panels::DIM).small());
                    });
                });
            });
    }

    fn act(&mut self, action: Action, session: &mut Session) {
        let status = session.timeline();
        let no_history = "no history to travel in: the Timeline panel's Keep history, or --history";
        match action {
            Action::Game { code, argument, echo } => {
                session.input.enqueue(GameEvent::Console { command: code as f32, argument });
                self.console.print(echo, panels::DIM);
            }
            Action::Tune(index, value) => session.input.enqueue(GameEvent::Tune { index, value }),
            Action::Tunable(name, value) => match self.tunables.command(name, value, session) {
                Ok(said) => self.console.print(said, panels::DIM),
                Err(error) => self.console.error(error),
            },
            Action::Bake => self.tunables.bake(session),
            Action::Help => self.console.help(),
            Action::Clear => self.console.clear(),
            Action::Diagnostics(None) => {
                let level = if self.shown.profiler { LEVELS[self.profiler.level].0 } else { "off" };
                self.console
                    .print(format!("diagnostics: {level}\nusage: /diagnostics off|low|medium|high"), panels::DIM);
            }
            Action::Diagnostics(Some(0)) => {
                self.shown.profiler = false;
                self.console.print("diagnostics: off", panels::DIM);
            }
            Action::Diagnostics(Some(level)) => {
                self.shown.profiler = true;
                self.profiler.level = level;
                self.console.print(format!("diagnostics: {}", LEVELS[level].0), panels::DIM);
            }
            Action::Pause => self.travel(Travel::Pause, session),
            Action::Play => self.travel(Travel::Play, session),
            Action::Step(count) => {
                for _ in 0..count {
                    session.step_frame();
                }
            }
            Action::Rewind(_) | Action::Replay(_) if !status.history => self.console.error(no_history),
            Action::Rewind(seconds) => self.travel(Travel::Back(seconds), session),
            Action::Replay(seconds) => self.travel(Travel::Replay(seconds), session),
            Action::Record(path) => {
                let path = match session.input_recording() {
                    Some(_) => None,
                    None => Some(path.map(PathBuf::from).unwrap_or_else(|| self.new_file("recordings", "penrec"))),
                };
                self.travel(Travel::Record(path), session);
            }
            Action::Shot(path) => {
                let path = path.map(PathBuf::from).unwrap_or_else(|| self.new_file("shots", "png"));
                self.console.print(format!("a shot with the overlay to {}", path.display()), panels::DIM);
                session.overlay_shot(path);
            }
        }
    }

    fn travel(&mut self, travel: Travel, session: &mut Session) {
        match travel {
            Travel::Pause => session.pause(),
            Travel::Play => session.resume(),
            Travel::Step => session.step_frame(),
            Travel::Back(seconds) => {
                let frame = session.timeline().frame;
                session.rewind_to(session.frames_before(frame, seconds));
            }
            Travel::To(frame) => session.rewind_to(frame),
            Travel::Replay(seconds) => session.replay_recent(seconds),
            Travel::KeepHistory => session.keep_history(),
            Travel::Record(Some(path)) => {
                session.record_input(Some(path));
                self.timeline.next_path(self.new_file("recordings", "penrec"));
            }
            Travel::Record(None) => session.record_input(None),
        }
    }
}

impl Overlay for DevOverlay {
    fn window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        // The backquote key is the overlay's alone, as in the browser.
        if let WindowEvent::KeyboardInput { event: key, .. } = event
            && key.physical_key == PhysicalKey::Code(KeyCode::Backquote)
        {
            if key.state == ElementState::Pressed && !key.repeat {
                self.toggle();
            }
            return false;
        }
        if !self.open {
            return true;
        }
        let _ = self.winit.on_window_event(window, event);
        let panels = self.over || self.context.egui_is_using_pointer();
        match event {
            WindowEvent::KeyboardInput { .. } | WindowEvent::Ime(_) => !self.typing,
            WindowEvent::CursorMoved { position, .. } => {
                let scale = self.context.pixels_per_point();
                let point = egui::pos2(position.x as f32 / scale, position.y as f32 / scale);
                self.over = self.context.layer_id_at(point).is_some_and(|layer| layer.order != egui::Order::Background);
                // A drag the game started stays the game's, over a panel or not.
                self.buttons != 0 || !(self.over || self.context.egui_is_using_pointer())
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } => {
                let game = self.buttons != 0 || !panels;
                if game {
                    self.buttons |= button_bit(*button);
                }
                game
            }
            // Releases always reach the game, as the browser's do.
            WindowEvent::MouseInput { state: ElementState::Released, button, .. } => {
                self.buttons &= !button_bit(*button);
                true
            }
            WindowEvent::MouseWheel { .. } | WindowEvent::Touch(_) => !panels,
            _ => true,
        }
    }

    fn update(&mut self, session: &mut Session, window: &Window) {
        let now = Instant::now();
        let interval = self.previous.replace(now).map_or(0.0, |previous| (now - previous).as_secs_f64() * 1000.0);
        self.read_log();
        let measuring = self.open && self.shown.profiler && self.profiler.measures();
        if measuring != self.measuring {
            session.measure_frames(measuring);
            self.measuring = measuring;
        }
        self.profiler.sample(interval, session.timeline().frame, session.frame_timings());
        if self.tunables.poll(session, self.open && self.shown.tunables) {
            self.console.tunables(&self.tunables);
        }
        if !self.open {
            self.painter.primitives.clear();
            self.accesskit = None;
            self.accesskit_requests.clear();
            return;
        }
        let mut input = self.winit.take_egui_input(window);
        input.events.extend(self.accesskit_requests.drain(..).map(egui::Event::AccessKitActionRequest));
        let (mut actions, mut travel) = (Vec::new(), Vec::new());
        let context = self.context.clone();
        let mut output = context.run_ui(input, |ui| self.layout(ui.ctx(), session, &mut actions, &mut travel));
        self.accesskit = output.platform_output.accesskit_update.take();
        self.winit.handle_platform_output(window, output.platform_output);
        self.painter.primitives = context.tessellate(output.shapes, output.pixels_per_point);
        self.painter.pixels_per_point = output.pixels_per_point;
        self.painter.textures.append(output.textures_delta);
        for action in actions {
            self.act(action, session);
        }
        for travel in travel {
            self.travel(travel, session);
        }
        // Keys held when a panel took the keyboard would never come up for
        // the game: it lets go of them, as when the window loses focus.
        let typing = context.egui_wants_keyboard_input();
        if typing && !self.typing {
            session.input.enqueue(GameEvent::FocusLost);
        }
        self.typing = typing;
    }

    fn paint(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) {
        self.painter.paint(device, queue, target);
    }

    fn wants_pointer(&self) -> bool {
        self.open && (self.over || self.context.egui_is_using_pointer())
    }

    fn accesskit_shown(&self) -> bool {
        self.open
    }

    fn accesskit_update(&mut self) -> Option<accesskit::TreeUpdate> {
        self.accesskit.take()
    }

    fn accesskit_action(&mut self, request: accesskit::ActionRequest) {
        if self.open {
            self.accesskit_requests.push(request);
        }
    }
}

/// Whether to show the overlay from the start: `--overlay`, or PEN_OVERLAY
/// set to anything but 0.
pub fn requested(flag: bool) -> bool {
    flag || std::env::var_os("PEN_OVERLAY").is_some_and(|value| value != "0")
}
