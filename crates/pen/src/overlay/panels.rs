//! The overlay's panels. Each keeps its own state and lays itself out in an
//! egui window; what they ask of the session comes back as `Action`s, which
//! `ui` carries out.
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use egui::text::{CCursor, CCursorRange};
use egui::{Color32, EventFilter, Key, Modifiers, RichText, Ui};
use pen_protocol::GameEvent;

use super::commands::{self, Action, Choice, Entry, LEVELS, Suggestion, Table};
use super::log::{self, Line, Source};
use super::tunables::{self, Declared};
use crate::game::FrameTimings;
use crate::session::{Session, TimelineStatus};

/// Lines the console keeps, as the browser's does.
const CONSOLE_LINES: usize = 400;
/// Commands the console's history keeps.
const HISTORY: usize = 100;
/// The console log's height, in points.
const LOG_HEIGHT: f32 = 180.0;

pub const COMMAND: Color32 = Color32::from_rgb(140, 170, 255);
pub const ERROR: Color32 = Color32::from_rgb(255, 110, 100);
pub const GOOD: Color32 = Color32::from_rgb(120, 220, 140);
pub const DIM: Color32 = Color32::from_rgb(150, 155, 165);
/// The frame time and CPU work, in the browser's graph colours.
const FRAME: Color32 = Color32::from_rgb(0x55, 0xe0, 0x6a);
const WORK: Color32 = Color32::from_rgb(0x39, 0xa9, 0xe6);

/// The colour of a kept line from `source`.
pub fn colour(source: Source, error: bool) -> Color32 {
    if error {
        return ERROR;
    }
    match source {
        Source::Game => Color32::from_rgb(225, 225, 225),
        Source::Build => Color32::from_rgb(230, 200, 110),
        Source::Reload => GOOD,
        Source::Asset => Color32::from_rgb(110, 200, 230),
        Source::Pipeline => Color32::from_rgb(200, 150, 240),
        Source::Session => Color32::from_rgb(175, 185, 205),
    }
}

fn ago(at: Instant) -> String {
    let seconds = at.elapsed().as_secs();
    match seconds {
        0..=59 => format!("{seconds} s ago"),
        60..=3599 => format!("{} min ago", seconds / 60),
        _ => format!("{} h ago", seconds / 3600),
    }
}

// ---------------------------------------------------------------------------
// Console

struct ConsoleLine {
    text: String,
    colour: Color32,
}

/// The browser console's counterpart: the game's commands with completion
/// (a slash opens the list; the arrows move through it, Tab takes the
/// highlighted choice, Enter also runs it when it needs nothing more, Escape
/// closes it) and history (the arrows, when no list is open), and the log.
pub struct Console {
    table: Table,
    catalog: Vec<Entry>,
    lines: VecDeque<ConsoleLine>,
    input: String,
    history: Vec<String>,
    /// Where the arrows are in the history, and the input before them.
    browsing: Option<(usize, String)>,
    suggestions: Vec<Suggestion>,
    active: usize,
    /// Take the keyboard on the next layout.
    pub focus: bool,
}

impl Console {
    pub fn new(table: Table) -> Console {
        let catalog = commands::catalog(&table);
        let mut console = Console {
            table,
            catalog,
            lines: VecDeque::new(),
            input: String::new(),
            history: Vec::new(),
            browsing: None,
            suggestions: Vec::new(),
            active: 0,
            focus: false,
        };
        let known = console.table.commands.len();
        console.print(
            format!("type /help for the controls and commands ({known} of the game's); ` hides this overlay"),
            DIM,
        );
        console
    }

    pub fn print(&mut self, text: impl Into<String>, colour: Color32) {
        self.lines.push_back(ConsoleLine { text: text.into(), colour });
        while self.lines.len() > CONSOLE_LINES {
            self.lines.pop_front();
        }
    }

    pub fn error(&mut self, text: impl Into<String>) {
        self.print(text, ERROR);
    }

    /// The tunables `/tune` completes, by name.
    pub fn tunables(&mut self, tunables: &Tunables) {
        if let Some(entry) = self.catalog.iter_mut().find(|entry| entry.name == "tune") {
            entry.options = tunables
                .list
                .iter()
                .map(|tunable| Choice { value: tunable.declared.name.clone(), about: tunable.declared.doc.clone() })
                .collect();
        }
    }

    pub fn clear(&mut self) {
        self.lines.clear();
    }

    pub fn take(&mut self, line: &Line) {
        self.print(line.text.clone(), colour(line.source, line.error));
    }

    /// `/help`: the controls (from the checkout's browser console) and the
    /// commands.
    pub fn help(&mut self) {
        let controls = &self.table.controls;
        if !controls.is_empty() {
            let width = controls.iter().map(|(keys, _)| keys.chars().count()).max().unwrap_or(0);
            let text = controls
                .iter()
                .map(|(keys, action)| if action.is_empty() { keys.clone() } else { format!("{keys:width$}  {action}") })
                .collect::<Vec<_>>()
                .join("\n");
            self.print(text, Color32::from_rgb(225, 225, 225));
        }
        let mut lines =
            vec!["the overlay: ` shows and hides it; Ctrl+Shift+P/period/comma/R travel in time".to_owned()];
        for command in &self.table.commands {
            let argument = match &command.argument {
                commands::Argument::None => String::new(),
                commands::Argument::Choice(choices) => {
                    format!(" {}", choices.iter().map(|choice| choice.value.as_str()).collect::<Vec<_>>().join("|"))
                }
                commands::Argument::Count { min, max, .. } => format!(" [{min}-{max}]"),
                commands::Argument::Number => " [NUMBER]".to_owned(),
            };
            lines.push(format!("/{}{argument}  {} (#{})", command.name, command.about, command.code));
        }
        for (name, argument, about) in commands::BUILT_IN {
            if self.table.command(name).is_none() {
                let argument = if argument.is_empty() { String::new() } else { format!(" {argument}") };
                lines.push(format!("/{name}{argument}  {about}"));
            }
        }
        self.print(lines.join("\n"), Color32::from_rgb(225, 225, 225));
    }

    fn refresh(&mut self) {
        self.suggestions = commands::suggest(&self.input, &self.catalog);
        self.active = 0;
    }

    fn close(&mut self) {
        self.suggestions.clear();
    }

    /// Runs a line: what it asks for, or why not.
    fn run(&mut self, text: &str, actions: &mut Vec<Action>) {
        self.print(format!("> {text}"), COMMAND);
        if self.history.last().is_none_or(|last| last != text) {
            self.history.push(text.to_owned());
            if self.history.len() > HISTORY {
                self.history.remove(0);
            }
        }
        match commands::parse(text, &self.table) {
            Ok(action) => actions.push(action),
            Err(message) => self.error(message),
        }
    }

    /// Takes `choice` into the input; with `submit`, runs it unless it still
    /// takes an argument, whose choices then open.
    fn choose(&mut self, index: usize, submit: bool, actions: &mut Vec<Action>) {
        let Some(choice) = self.suggestions.get(index).cloned() else { return };
        self.input = choice.text.clone();
        if submit && choice.complete {
            self.input.clear();
            self.close();
            self.run(choice.text.trim(), actions);
        } else {
            self.refresh();
        }
    }

    /// The previous (`back`) or next command in the history.
    fn recall(&mut self, back: bool) {
        let (index, draft) = match self.browsing.take() {
            Some((index, draft)) => (Some(index), draft),
            None => (None, self.input.clone()),
        };
        let next = match (index, back) {
            (None, true) => self.history.len().checked_sub(1),
            (None, false) => None,
            (Some(index), true) => Some(index.saturating_sub(1)),
            (Some(index), false) => (index + 1 < self.history.len()).then_some(index + 1),
        };
        match next {
            Some(index) => {
                self.input = self.history[index].clone();
                self.browsing = Some((index, draft));
            }
            None => self.input = draft,
        }
        self.close();
    }

    pub fn show(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        let id = ui.make_persistent_id("console input");
        let mut moved_cursor = false;
        if ui.memory(|memory| memory.has_focus(id)) {
            let pressed = |key| ui.input_mut(|input| input.consume_key(Modifiers::NONE, key));
            let (up, down, tab, enter, escape) = (
                pressed(Key::ArrowUp),
                pressed(Key::ArrowDown),
                pressed(Key::Tab),
                pressed(Key::Enter),
                pressed(Key::Escape),
            );
            let open = !self.suggestions.is_empty();
            if (up || down) && open {
                let count = self.suggestions.len();
                self.active = (self.active + if down { 1 } else { count - 1 }) % count;
            } else if up || down {
                self.recall(up);
                moved_cursor = true;
            }
            if tab {
                if open {
                    self.choose(self.active, false, actions);
                } else {
                    self.refresh();
                }
                moved_cursor = true;
            }
            if enter {
                if open {
                    self.choose(self.active, true, actions);
                } else {
                    let text = std::mem::take(&mut self.input);
                    let text = text.trim();
                    if !text.is_empty() {
                        self.run(text, actions);
                    }
                }
                self.browsing = None;
                moved_cursor = true;
            }
            if escape {
                if open {
                    self.close();
                } else {
                    ui.memory_mut(|memory| memory.surrender_focus(id));
                }
            }
        }

        egui::ScrollArea::vertical().auto_shrink(false).stick_to_bottom(true).max_height(LOG_HEIGHT).show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 2.0;
            for line in &self.lines {
                ui.add(egui::Label::new(RichText::new(&line.text).monospace().color(line.colour)).wrap());
            }
        });

        if !self.suggestions.is_empty() {
            let mut chosen = None;
            egui::Frame::new().fill(Color32::from_black_alpha(120)).inner_margin(4.0).show(ui, |ui| {
                egui::ScrollArea::vertical().id_salt("console choices").max_height(132.0).show(ui, |ui| {
                    for (index, suggestion) in self.suggestions.iter().enumerate() {
                        let mut label = egui::text::LayoutJob::default();
                        let (typed, rest) = suggestion.label.split_at(suggestion.matched.min(suggestion.label.len()));
                        let mono = egui::TextFormat::simple(egui::FontId::monospace(13.0), Color32::WHITE);
                        label.append(typed, 0.0, egui::TextFormat { color: COMMAND, ..mono.clone() });
                        label.append(rest, 0.0, mono);
                        label.append(
                            &format!("   {}", suggestion.about),
                            0.0,
                            egui::TextFormat::simple(egui::FontId::proportional(12.0), DIM),
                        );
                        let response = ui.selectable_label(index == self.active, label);
                        if index == self.active {
                            response.scroll_to_me(None);
                        }
                        if response.clicked() {
                            chosen = Some(index);
                        }
                    }
                });
            });
            if let Some(index) = chosen {
                self.choose(index, true, actions);
                self.focus = true;
                moved_cursor = true;
            }
        }

        let response = ui.add(
            egui::TextEdit::singleline(&mut self.input)
                .id(id)
                .font(egui::TextStyle::Monospace)
                .desired_width(f32::INFINITY)
                .hint_text("/help")
                .return_key(None::<egui::KeyboardShortcut>)
                .event_filter(EventFilter { tab: true, horizontal_arrows: true, vertical_arrows: true, escape: true }),
        );
        if response.changed() {
            self.browsing = None;
            self.refresh();
        }
        if std::mem::take(&mut self.focus) {
            response.request_focus();
        }
        if moved_cursor && let Some(mut state) = egui::text_edit::TextEditState::load(ui.ctx(), id) {
            let end = CCursor::new(self.input.chars().count());
            state.cursor.set_char_range(Some(CCursorRange::one(end)));
            state.store(ui.ctx(), id);
        }
    }
}

// ---------------------------------------------------------------------------
// Profiler

/// Steps the graph shows.
const GRAPH_STEPS: usize = 240;
/// Frames the phases are averaged over.
const PHASE_FRAMES: usize = 30;

struct Step {
    interval_ms: f64,
    /// The frame's CPU work, when one ran and was measured.
    work_ms: Option<f64>,
}

/// What the numbers show, refreshed a few times a second so they can be read.
#[derive(Default)]
struct Summary {
    fps: f64,
    frame_ms: f64,
    p95_ms: f64,
    max_ms: f64,
    work_ms: Option<f64>,
    phases: Option<FrameTimings>,
}

/// The browser's diagnostics overlay: frame interval and CPU work with a
/// graph, the frame's phases, the renderer's draws, guest memory and the GPU.
pub struct Profiler {
    /// An index into `LEVELS` (1 low, 2 medium, 3 high).
    pub level: usize,
    steps: VecDeque<Step>,
    timings: VecDeque<FrameTimings>,
    /// The frame whose timings were taken last.
    last_frame: Option<u64>,
    summary: Summary,
    refreshed: Option<Instant>,
}

impl Profiler {
    pub fn new() -> Profiler {
        Profiler {
            level: 3,
            steps: VecDeque::new(),
            timings: VecDeque::new(),
            last_frame: None,
            summary: Summary::default(),
            refreshed: None,
        }
    }

    /// Whether frames need timing: the CPU work and phases need them.
    pub fn measures(&self) -> bool {
        self.level >= 2
    }

    /// A step `interval_ms` after the last; `frame` is the session's
    /// position, so a paused step adds no work.
    pub fn sample(&mut self, interval_ms: f64, frame: u64, timings: Option<FrameTimings>) {
        let ran = self.last_frame != Some(frame);
        self.last_frame = Some(frame);
        let timings = timings.filter(|_| ran);
        self.steps.push_back(Step { interval_ms, work_ms: timings.map(|timings| timings.total_ms) });
        if self.steps.len() > GRAPH_STEPS {
            self.steps.pop_front();
        }
        if let Some(timings) = timings {
            self.timings.push_back(timings);
            if self.timings.len() > PHASE_FRAMES {
                self.timings.pop_front();
            }
        }
        if self.refreshed.is_none_or(|at| at.elapsed() >= Duration::from_millis(250)) {
            self.refreshed = Some(Instant::now());
            self.summarise();
        }
    }

    fn summarise(&mut self) {
        // The last second of steps.
        let mut recent = Vec::new();
        let mut time = 0.0;
        for step in self.steps.iter().rev() {
            if time >= 1000.0 {
                break;
            }
            time += step.interval_ms;
            recent.push(step);
        }
        if recent.is_empty() {
            return;
        }
        let mut intervals: Vec<f64> = recent.iter().map(|step| step.interval_ms).collect();
        intervals.sort_by(f64::total_cmp);
        let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let work: Vec<f64> = recent.iter().filter_map(|step| step.work_ms).collect();
        let phases = (!self.timings.is_empty()).then(|| {
            let count = self.timings.len() as f64;
            let mean = |pick: fn(&FrameTimings) -> f64| self.timings.iter().map(pick).sum::<f64>() / count;
            FrameTimings {
                guest_ms: mean(|t| t.guest_ms),
                wait_ms: mean(|t| t.wait_ms),
                decode_ms: mean(|t| t.decode_ms),
                validate_ms: mean(|t| t.validate_ms),
                submit_ms: mean(|t| t.submit_ms),
                housekeeping_ms: mean(|t| t.housekeeping_ms),
                total_ms: mean(|t| t.total_ms),
                packet_bytes: (self.timings.iter().map(|t| t.packet_bytes as f64).sum::<f64>() / count) as usize,
            }
        });
        self.summary = Summary {
            fps: if mean > 0.0 { 1000.0 / mean } else { 0.0 },
            frame_ms: mean,
            p95_ms: intervals[(intervals.len() * 95 / 100).min(intervals.len() - 1)],
            max_ms: intervals[intervals.len() - 1],
            work_ms: (!work.is_empty()).then(|| work.iter().sum::<f64>() / work.len() as f64),
            phases,
        };
    }

    /// The frame interval (green) and CPU work (blue) of the last steps,
    /// with a dashed line at 60 fps, as the browser draws them.
    fn graph(&self, ui: &mut Ui) {
        let size = egui::vec2(ui.available_width(), 48.0);
        let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, Color32::from_black_alpha(90));
        let ceiling =
            self.steps.iter().map(|step| step.interval_ms.max(step.work_ms.unwrap_or(0.0)) * 1.15).fold(20.0, f64::max);
        let y = |value: f64| rect.bottom() - 1.0 - (value / ceiling) as f32 * (rect.height() - 3.0);
        let target = y(1000.0 / 60.0);
        painter.extend(egui::Shape::dashed_line(
            &[egui::pos2(rect.left(), target), egui::pos2(rect.right(), target)],
            egui::Stroke::new(1.0, Color32::from_white_alpha(64)),
            2.0,
            3.0,
        ));
        let x = |index: usize| {
            rect.right() - (self.steps.len() - 1 - index) as f32 * rect.width() / (GRAPH_STEPS - 1) as f32
        };
        let work: Vec<egui::Pos2> = self
            .steps
            .iter()
            .enumerate()
            .filter_map(|(index, step)| Some(egui::pos2(x(index), y(step.work_ms?))))
            .collect();
        painter.line(work, egui::Stroke::new(1.5, WORK));
        let frames =
            self.steps.iter().enumerate().map(|(index, step)| egui::pos2(x(index), y(step.interval_ms))).collect();
        painter.line(frames, egui::Stroke::new(1.5, FRAME));
        painter.text(
            rect.left_top() + egui::vec2(4.0, 2.0),
            egui::Align2::LEFT_TOP,
            format!("0–{:.0} ms", ceiling.ceil()),
            egui::FontId::proportional(11.0),
            DIM,
        );
    }

    pub fn show(&mut self, ui: &mut Ui, details: &Details) {
        ui.spacing_mut().item_spacing.y = 2.0;
        ui.horizontal(|ui| {
            for (index, (name, about)) in LEVELS.iter().enumerate().skip(1) {
                ui.selectable_value(&mut self.level, index, *name).on_hover_text(*about);
            }
        });
        let summary = &self.summary;
        let row = |ui: &mut Ui, label: &str, colour: Color32, value: String| {
            ui.label(RichText::new(label).strong().color(colour));
            ui.add(egui::Label::new(RichText::new(value).monospace()).wrap());
            ui.end_row();
        };
        egui::Grid::new("profiler numbers").num_columns(2).spacing([10.0, 2.0]).show(ui, |ui| {
            row(ui, "FPS", FRAME, format!("{:.0}  ({:.2} ms a frame)", summary.fps, summary.frame_ms));
            if self.level >= 2 {
                row(ui, "FRAME", FRAME, format!("p95 {:.2}  max {:.2} ms", summary.p95_ms, summary.max_ms));
                let work = summary.work_ms.map_or("—".to_owned(), |work| format!("{work:.2} ms"));
                row(ui, "CPU", WORK, work);
            }
        });
        if self.level < 2 {
            return;
        }
        self.graph(ui);
        ui.horizontal(|ui| {
            ui.label(RichText::new("— frame interval").color(FRAME).small());
            ui.label(RichText::new("— CPU work").color(WORK).small());
            ui.label(RichText::new(format!("· last {GRAPH_STEPS} steps")).color(DIM).small());
        });
        if self.level < 3 {
            return;
        }
        ui.separator();
        match &summary.phases {
            Some(phases) => {
                let total = phases.total_ms.max(1e-6);
                let mut rows = vec![("guest", phases.guest_ms)];
                if details.timeline.pipelined {
                    rows.push(("waited", phases.wait_ms));
                }
                rows.extend([
                    ("decode", phases.decode_ms),
                    ("validate", phases.validate_ms),
                    ("render+present", phases.submit_ms),
                    ("housekeeping", phases.housekeeping_ms),
                ]);
                egui::Grid::new("profiler phases").num_columns(3).spacing([8.0, 1.0]).show(ui, |ui| {
                    for (name, ms) in rows {
                        ui.label(RichText::new(name).small());
                        let bar = egui::ProgressBar::new((ms / total) as f32).desired_width(100.0).desired_height(7.0);
                        ui.add(bar);
                        ui.label(RichText::new(format!("{ms:6.2} ms")).monospace().small());
                        ui.end_row();
                    }
                });
            }
            None => {
                ui.label(RichText::new("phases: waiting for a frame to run (paused?)").color(DIM).small());
            }
        }
        ui.separator();
        let stats = details.stats;
        egui::Grid::new("profiler details").num_columns(2).spacing([10.0, 2.0]).show(ui, |ui| {
            let draws = format!(
                "{} ({} instanced), {} instances",
                stats.draw_calls, stats.instanced_draw_calls, stats.instances
            );
            row(ui, "DRAWS", Color32::from_rgb(230, 200, 110), draws);
            let memory =
                details.memory.map_or("—".to_owned(), |bytes| format!("{:.1} MiB", bytes as f64 / 1048576.0));
            let packet = summary
                .phases
                .as_ref()
                .map_or("—".to_owned(), |phases| format!("{:.1} KiB", phases.packet_bytes as f64 / 1024.0));
            row(ui, "MEM", Color32::from_rgb(200, 150, 240), format!("guest {memory}, packet {packet}"));
            row(ui, "RES", DIM, format!("{} × {}", details.size.0, details.size.1));
            ui.label(RichText::new("GPU").strong().color(Color32::from_rgb(255, 150, 90)));
            ui.add(egui::Label::new(RichText::new(&details.adapter).small()).wrap());
            ui.end_row();
        });
    }
}

/// What the profiler shows besides its own samples.
pub struct Details {
    pub stats: pen_render::RenderStats,
    pub memory: Option<usize>,
    pub adapter: String,
    pub size: (u32, u32),
    pub timeline: TimelineStatus,
}

// ---------------------------------------------------------------------------
// Timeline

/// What the timeline panel asks for.
pub enum Travel {
    Pause,
    Play,
    Step,
    /// Back this many seconds of game time.
    Back(f64),
    To(u64),
    Replay(f64),
    KeepHistory,
    Record(Option<PathBuf>),
}

/// Play and pause, step, rewind (by a second, a frame, or to a frame on the
/// scrub slider), replay the last seconds, and record the input.
pub struct TimelinePanel {
    replay_seconds: f64,
    /// Where the next recording goes.
    path: String,
    /// The slider's frame while it is dragged.
    scrub: Option<u64>,
}

impl TimelinePanel {
    pub fn new(path: PathBuf) -> TimelinePanel {
        TimelinePanel { replay_seconds: crate::session::REPLAY_SECONDS, path: path.display().to_string(), scrub: None }
    }

    /// A new default path for the next recording, once one has been made.
    pub fn next_path(&mut self, path: PathBuf) {
        self.path = path.display().to_string();
    }

    pub fn show(&mut self, ui: &mut Ui, status: &TimelineStatus, recording: Option<(String, u64)>) -> Vec<Travel> {
        let mut travel = Vec::new();
        let state = match (status.paused, status.replaying) {
            (true, true) => RichText::new("paused in a replay").color(Color32::from_rgb(255, 200, 90)),
            (true, false) => RichText::new("paused").color(Color32::from_rgb(255, 200, 90)),
            (false, true) => RichText::new("replaying").color(COMMAND),
            (false, false) => RichText::new("playing").color(GOOD),
        };
        ui.horizontal(|ui| {
            ui.label(state.strong());
            ui.label(RichText::new(format!("frame {}", status.frame)).monospace());
            if status.latest > status.frame {
                ui.label(RichText::new(format!("({} recorded ahead)", status.latest - status.frame)).color(DIM));
            }
            if status.pipelined {
                ui.label(RichText::new("overlapped").color(DIM));
            }
        });
        ui.horizontal(|ui| {
            if status.paused {
                if ui.button("Play").on_hover_text("Ctrl+Shift+P; after a rewind, live from here").clicked() {
                    travel.push(Travel::Play);
                }
            } else if ui.button("Pause").on_hover_text("Ctrl+Shift+P").clicked() {
                travel.push(Travel::Pause);
            }
            if ui.button("Step").on_hover_text("one frame (Ctrl+Shift+period)").clicked() {
                travel.push(Travel::Step);
            }
            ui.add_enabled_ui(status.history, |ui| {
                if ui.button("Back a frame").on_hover_text("Ctrl+Shift+comma").clicked() {
                    travel.push(Travel::To(status.frame.saturating_sub(1)));
                }
                if ui.button("Back 1 s").clicked() {
                    travel.push(Travel::Back(1.0));
                }
            });
        });
        match status.earliest.filter(|_| status.history) {
            Some(earliest) => {
                ui.label(
                    RichText::new(format!("history: {:.1} s, frames {earliest} to {}", status.seconds, status.latest))
                        .color(DIM),
                );
                let mut frame = self.scrub.unwrap_or(status.frame).clamp(earliest, status.latest.max(earliest));
                let slider =
                    egui::Slider::new(&mut frame, earliest..=status.latest.max(earliest)).text("frame").integer();
                let response = ui.add_sized([ui.available_width(), 18.0], slider);
                if response.dragged() || response.changed() {
                    self.scrub = Some(frame);
                    if response.changed() {
                        travel.push(Travel::To(frame));
                    }
                }
                if !response.dragged() {
                    self.scrub = None;
                }
                ui.horizontal(|ui| {
                    if ui.button("Replay the last").on_hover_text("then play on live (Ctrl+Shift+R: 5 s)").clicked() {
                        travel.push(Travel::Replay(self.replay_seconds));
                    }
                    ui.add(egui::DragValue::new(&mut self.replay_seconds).range(0.1..=60.0).speed(0.1).suffix(" s"));
                });
            }
            None => {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("No history to rewind in.").color(DIM));
                    if ui
                        .button("Keep history")
                        .on_hover_text("the last 10 s from now on (pen dev always does)")
                        .clicked()
                    {
                        travel.push(Travel::KeepHistory);
                    }
                });
            }
        }
        ui.separator();
        match recording {
            Some((path, from)) => {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("recording").color(ERROR).strong());
                    let frames = status.frame.saturating_sub(from);
                    ui.label(RichText::new(format!("{frames} frames since frame {from}")).monospace());
                    if ui.button("Stop").clicked() {
                        travel.push(Travel::Record(None));
                    }
                });
                ui.add(egui::Label::new(RichText::new(path).monospace().small().color(DIM)).wrap());
            }
            None => {
                ui.horizontal(|ui| {
                    if ui.button("Record input").on_hover_text("from this frame, to replay with --replay").clicked() {
                        travel.push(Travel::Record(Some(PathBuf::from(self.path.trim()))));
                    }
                    ui.add(
                        egui::TextEdit::singleline(&mut self.path)
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY),
                    );
                });
            }
        }
        travel
    }
}

// ---------------------------------------------------------------------------
// Hot reload

enum Build {
    Waiting,
    Compiled(String),
    Failed(String),
}

struct Event {
    at: Instant,
    text: String,
    colour: Color32,
}

/// The Blot build's status and errors (the compiler's output, `log`), the
/// last new build's fate, and asset and recipe events.
pub struct Reload {
    developing: bool,
    build: Build,
    /// The failed build's diagnostics.
    errors: Vec<String>,
    /// The last word on a new build of the running game.
    runtime: Option<(Instant, String, bool)>,
    events: VecDeque<Event>,
}

impl Reload {
    pub fn new(developing: bool) -> Reload {
        Reload { developing, build: Build::Waiting, errors: Vec::new(), runtime: None, events: VecDeque::new() }
    }

    fn event(&mut self, line: &Line) {
        self.events.push_back(Event { at: line.at, text: line.text.clone(), colour: colour(line.source, line.error) });
        if self.events.len() > 60 {
            self.events.pop_front();
        }
    }

    pub fn take(&mut self, line: &Line) {
        match line.source {
            Source::Build if line.text.starts_with("blot: ") => {
                let status = line.text.trim_start_matches("blot: ").to_owned();
                self.build = if line.error { Build::Failed(status) } else { Build::Compiled(status) };
                self.errors.clear();
                self.event(line);
            }
            Source::Build if line.error => self.errors.push(line.text.clone()),
            Source::Reload => {
                // The new build's fate, not hints about it.
                if line.error || line.text.starts_with("Blot reload") || line.text.starts_with("dev: compiled") {
                    self.runtime = Some((line.at, line.text.clone(), line.error));
                }
                self.event(line);
            }
            Source::Asset | Source::Pipeline => self.event(line),
            _ => {}
        }
    }

    /// The build is broken: the toolbar says so.
    pub fn failing(&self) -> bool {
        matches!(self.build, Build::Failed(_))
    }

    pub fn show(&mut self, ui: &mut Ui) {
        if !self.developing {
            ui.label(
                RichText::new("Hot reload is off: `pen dev` reloads the program, shaders and assets as they change.")
                    .color(DIM),
            );
        }
        ui.horizontal(|ui| {
            ui.label(RichText::new("Blot build").strong());
            match &self.build {
                Build::Waiting => ui.label(RichText::new("no build reported yet").color(DIM)),
                Build::Compiled(status) => ui.label(RichText::new(status).color(GOOD)),
                Build::Failed(status) => ui.add(egui::Label::new(RichText::new(status).color(ERROR)).wrap()),
            };
        });
        if !self.errors.is_empty() {
            egui::ScrollArea::vertical().id_salt("build errors").max_height(180.0).show(ui, |ui| {
                ui.add(
                    egui::Label::new(RichText::new(self.errors.join("\n")).monospace().small().color(ERROR))
                        .wrap()
                        .selectable(true),
                );
            });
        }
        if let Some((at, text, error)) = &self.runtime {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Running build").strong());
                ui.add(egui::Label::new(RichText::new(text).color(if *error { ERROR } else { GOOD })).wrap());
                ui.label(RichText::new(ago(*at)).color(DIM).small());
            });
        }
        ui.separator();
        ui.label(RichText::new("Builds, assets and recipes").strong());
        egui::ScrollArea::vertical().id_salt("reload events").max_height(200.0).stick_to_bottom(true).show(ui, |ui| {
            if self.events.is_empty() {
                ui.label(RichText::new("nothing yet").color(DIM));
            }
            for event in &self.events {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(ago(event.at)).color(DIM).small().monospace());
                    ui.label(RichText::new(&event.text).color(event.colour).small());
                });
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Tunables

/// How long a value sent may go unseen in the game's state, while it plays,
/// before the panel takes the state's word instead (a rewind or a load
/// undid it).
const UNSEEN: Duration = Duration::from_secs(2);
/// Frames between readings of the game's state while the panel shows.
const OBSERVE_EVERY: u32 = 15;
/// How long the declarations file must be quiet before it is read again.
const SETTLE: Duration = Duration::from_millis(30);

/// Watches the checkout's `src` for its declarations file changing.
struct Watch {
    _watcher: notify::RecommendedWatcher,
    events: std::sync::mpsc::Receiver<notify::Result<notify::Event>>,
    file: PathBuf,
    changed: Option<Instant>,
}

impl Watch {
    fn new(root: &Path) -> Option<Watch> {
        use notify::Watcher;
        let file = tunables::path(root);
        let (sender, events) = std::sync::mpsc::channel();
        let mut watcher = notify::recommended_watcher(sender).ok()?;
        let directory = file.parent()?;
        if let Err(error) = watcher.watch(directory, notify::RecursiveMode::NonRecursive) {
            eprintln!("tunables: cannot watch {}: {error}", directory.display());
            return None;
        }
        Some(Watch { _watcher: watcher, events, file, changed: None })
    }

    /// Whether the file changed and has settled since the last call.
    fn changed(&mut self) -> bool {
        let name = self.file.file_name();
        while let Ok(event) = self.events.try_recv() {
            if let Ok(event) = event
                && !matches!(event.kind, notify::EventKind::Access(_))
                && event.paths.iter().any(|path| path.file_name() == name)
            {
                self.changed = Some(Instant::now());
            }
        }
        if self.changed.is_some_and(|at| at.elapsed() >= SETTLE) {
            self.changed = None;
            return true;
        }
        false
    }
}

/// A live-tunable constant (gdev's README, "Tunables") as the panel shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct Tunable {
    pub declared: Declared,
    /// What the game holds: as its state last said, or as sent since.
    pub value: f32,
    /// A value sent that the game's state does not show yet, and when.
    pending: Option<(f32, Instant)>,
}

impl Tunable {
    pub fn tuned(&self) -> bool {
        self.value != self.declared.default
    }
}

/// The Tunables panel: a slider for each of the checkout's tunables, whose
/// value can also be typed (click it), a reset to its default, and Bake,
/// which writes the game's values into `src/tunables.json` as the defaults.
/// Changes reach the game as Tune events (`Action::Tune`), and the panel
/// follows the game's state (`observe`), so a load, rewind, reload or replay
/// shows in it.
pub struct Tunables {
    /// The checkout.
    root: Option<PathBuf>,
    pub list: Vec<Tunable>,
    /// Why the file does not load, if it does not.
    pub problem: Option<String>,
    /// What the last bake did.
    status: Option<(String, Color32)>,
    /// A bake under way, on a thread of its own: what it says, once done.
    baking: Option<std::sync::mpsc::Receiver<Result<String, String>>>,
    watch: Option<Watch>,
    /// Frames since the game's state was last read; none until it is first.
    observed: Option<u32>,
}

/// Decimals a slider shows: the step's, or a few.
fn decimals(step: Option<f32>) -> usize {
    step.map_or(4, |step| format!("{step}").split('.').nth(1).map_or(0, str::len))
}

impl Tunables {
    /// The checkout's tunables (`src/tunables.json`), at their defaults until
    /// the game's state says otherwise.
    pub fn load(root: Option<&Path>) -> Tunables {
        let mut tunables = Tunables {
            root: root.map(Path::to_path_buf),
            list: Vec::new(),
            problem: None,
            status: None,
            baking: None,
            watch: root.and_then(Watch::new),
            observed: None,
        };
        tunables.reload();
        tunables
    }

    /// Reads the file again: tunables still declared keep their values, new
    /// ones start at their defaults. A file that does not load leaves the
    /// list as it was and says why.
    pub fn reload(&mut self) {
        let Some(root) = &self.root else { return };
        match tunables::load(root) {
            Ok(declared) => {
                self.problem = None;
                let previous = std::mem::take(&mut self.list);
                self.list = declared
                    .into_iter()
                    .map(|declared| match previous.iter().find(|kept| kept.declared.id == declared.id) {
                        Some(kept) => {
                            Tunable { value: kept.value.clamp(declared.min, declared.max), pending: kept.pending, declared }
                        }
                        None => Tunable { value: declared.default, pending: None, declared },
                    })
                    .collect();
            }
            Err(problem) => self.problem = Some(problem),
        }
    }

    pub fn index(&self, name: &str) -> Option<usize> {
        self.list.iter().position(|tunable| tunable.declared.name == name)
    }

    /// The game's word on each tunable, from what its state holds (by id);
    /// a value sent shows once the game holds it. While the game is paused,
    /// values sent wait for it to run.
    pub fn observe(&mut self, tuned: &HashMap<u32, f32>, paused: bool) {
        for tunable in &mut self.list {
            let live = tunable.declared.live(tuned);
            tunable.pending = match tunable.pending {
                Some((sent, _)) if paused && sent != live => Some((sent, Instant::now())),
                Some((sent, at)) if sent != live && at.elapsed() < UNSEEN => Some((sent, at)),
                _ => None,
            };
            if tunable.pending.is_none() {
                tunable.value = live;
            }
        }
    }

    /// Sets tunable `index` to `value`, held in its range, and returns the
    /// Tune to send the game: its id and the value.
    pub fn set(&mut self, index: usize, value: f32) -> (u32, f32) {
        let tunable = &mut self.list[index];
        let value = value.clamp(tunable.declared.min, tunable.declared.max);
        tunable.value = value;
        tunable.pending = Some((value, Instant::now()));
        (tunable.declared.id, value)
    }

    /// The values to bake: each tunable's that is not its default.
    pub fn tuned(&self) -> Vec<(String, f32)> {
        self.list.iter().filter(|tunable| tunable.tuned()).map(|t| (t.declared.name.clone(), t.value)).collect()
    }

    /// Reads what the game holds now.
    fn refresh(&mut self, session: &mut Session) {
        if let Some(saved) = session.snapshot() {
            self.observe(&tunables::tuned_values(&saved.state), session.timeline().paused);
            self.observed = Some(0);
        }
    }

    /// Once a frame: reads the declarations again when the file changed, the
    /// game's state every few frames while the panel `shows`, and a bake's
    /// outcome. Returns whether the declarations changed.
    pub fn poll(&mut self, session: &mut Session, shows: bool) -> bool {
        let changed = self.watch.as_mut().is_some_and(Watch::changed);
        if changed {
            let before: Vec<Declared> = self.list.iter().map(|tunable| tunable.declared.clone()).collect();
            self.reload();
            match &self.problem {
                Some(problem) => log::error(Source::Session, format!("tunables: {problem}")),
                None if self.list.iter().map(|tunable| &tunable.declared).ne(before.iter()) => {
                    log::info(Source::Session, format!("tunables: read {} from src/tunables.json", self.list.len()))
                }
                None => {}
            }
        }
        match self.observed {
            _ if !shows => self.observed = None,
            Some(frames) if frames + 1 < OBSERVE_EVERY => self.observed = Some(frames + 1),
            _ => self.refresh(session),
        }
        if let Some(result) = self.baking.as_ref().and_then(|baking| baking.try_recv().ok()) {
            self.baking = None;
            match &result {
                Ok(said) => log::info(Source::Session, format!("tunables: baked into src/tunables.json\n{said}")),
                Err(error) => log::error(Source::Session, format!("tunables: {error}")),
            }
            self.status = Some(match result {
                Ok(said) => (said, GOOD),
                Err(error) => (error, ERROR),
            });
        }
        changed
    }

    /// Writes the game's values into the checkout's `src/tunables.json` as
    /// the defaults, on a thread of its own (it runs Deno); `poll` says how
    /// it went. The game keeps its values, which are now the defaults, so
    /// nothing in it moves when the new build loads.
    pub fn bake(&mut self, session: &mut Session) {
        let Some(root) = self.root.clone() else { return };
        if self.baking.is_some() {
            return;
        }
        self.refresh(session);
        let values = self.tuned();
        if values.is_empty() {
            self.status = Some(("nothing to bake: each value is its default".to_owned(), DIM));
            return;
        }
        let (sender, receiver) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new().name("pen-bake".into()).spawn(move || {
            let _ = sender.send(tunables::bake(&root, &values).map_err(|error| format!("{error:#}")));
        });
        match spawned {
            Ok(_) => self.baking = Some(receiver),
            Err(error) => self.status = Some((format!("cannot bake: {error}"), ERROR)),
        }
    }

    /// `/tune [NAME [VALUE]]`: the tunables, one of them, or a new value
    /// for it, sent to the game; what to print, or why not.
    pub fn command(&mut self, name: Option<String>, value: Option<f32>, session: &mut Session) -> Result<String, String> {
        self.refresh(session);
        let Some(name) = name else { return Ok(self.listing()) };
        let Some(index) = self.index(&name) else {
            return Err(format!("no tunable {name}; /tune lists them"));
        };
        let Some(wanted) = value else {
            let Tunable { declared, value, .. } = &self.list[index];
            let about = if declared.doc.is_empty() { String::new() } else { format!("\n{}", declared.doc) };
            return Ok(format!(
                "{name} = {value} (default {}, {} to {}){about}",
                declared.default, declared.min, declared.max
            ));
        };
        let (id, value) = self.set(index, wanted);
        session.input.enqueue(GameEvent::Tune { index: id, value });
        Ok(if value == wanted { format!("{name}: {value}") } else { format!("{name}: {value} (held in its range)") })
    }

    /// The `/tune` listing: each tunable, its value, default and range.
    pub fn listing(&self) -> String {
        if self.list.is_empty() {
            return "no tunables: declare them in src/tunables.json".to_owned();
        }
        let width = self.list.iter().map(|tunable| tunable.declared.name.len()).max().unwrap_or(0);
        self.list
            .iter()
            .map(|tunable| {
                let Declared { name, default, min, max, .. } = &tunable.declared;
                let mark = if tunable.tuned() { "*" } else { " " };
                format!("{mark}{name:width$}  {}  (default {default}, {min} to {max})", tunable.value)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Lays the panel out; what it asks for goes into `actions`.
    pub fn show(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        if let Some(problem) = &self.problem {
            ui.add(egui::Label::new(RichText::new(problem).color(ERROR)).wrap());
        }
        if self.list.is_empty() {
            let text = match &self.root {
                Some(_) => "No tunables: declare them in src/tunables.json (gdev's README, \"Tunables\").",
                None => "No tunables: there is no checkout.",
            };
            ui.add(egui::Label::new(RichText::new(text).color(DIM)).wrap());
            return;
        }
        let mut changes = Vec::new();
        egui::Grid::new("tunables").num_columns(3).spacing([8.0, 4.0]).show(ui, |ui| {
            for (index, tunable) in self.list.iter().enumerate() {
                let Declared { name, default, min, max, step, doc, .. } = &tunable.declared;
                let label = RichText::new(name).monospace().color(if tunable.tuned() { COMMAND } else { DIM });
                let about = if doc.is_empty() { String::new() } else { format!("{doc}\n") };
                ui.label(label).on_hover_text(format!("{about}default {default}, {min} to {max}"));
                let mut value = tunable.value;
                let mut slider = egui::Slider::new(&mut value, *min..=*max).fixed_decimals(decimals(*step));
                if let Some(step) = step {
                    slider = slider.step_by(f64::from(*step));
                }
                if ui.add(slider).changed() {
                    changes.push((index, value));
                }
                let reset = ui
                    .add_enabled(tunable.tuned(), egui::Button::new("reset").small())
                    .on_hover_text(format!("back to {default}"));
                if reset.clicked() {
                    changes.push((index, *default));
                }
                ui.end_row();
            }
        });
        for (index, value) in changes {
            let (id, value) = self.set(index, value);
            actions.push(Action::Tune(id, value));
        }
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            let tuned = self.list.iter().filter(|tunable| tunable.tuned()).count();
            let bake = ui
                .add_enabled(self.baking.is_none() && tuned > 0 && self.root.is_some(), egui::Button::new("Bake"))
                .on_hover_text("Write the game's values into src/tunables.json as the defaults");
            if bake.clicked() {
                actions.push(Action::Bake);
            }
            match &self.status {
                _ if self.baking.is_some() => ui.label(RichText::new("baking...").color(DIM)),
                Some((text, colour)) => ui.label(RichText::new(text).color(*colour).small()),
                None => ui.label(RichText::new(format!("{tuned} tuned")).color(DIM)),
            };
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(source: Source, error: bool, text: &str) -> Line {
        Line { serial: 0, source, error, text: text.to_owned(), at: Instant::now() }
    }

    #[test]
    fn the_tunables_panel_follows_the_game() {
        let declared = |name: &str, default: f32| Declared {
            name: name.into(),
            id: tunables::tunable_id(name),
            default,
            min: 0.0,
            max: 4.0,
            step: Some(0.5),
            doc: String::new(),
        };
        let mut panel = Tunables::load(None);
        panel.list = ["a.speed", "b.wait"]
            .map(|name| Tunable { declared: declared(name, 1.0), value: 1.0, pending: None })
            .into();
        let id = panel.list[0].declared.id;
        // Held in range, sent, and shown until the game's state has it.
        assert_eq!(panel.set(0, 9.0), (id, 4.0));
        panel.observe(&HashMap::new(), false);
        assert_eq!(panel.list[0].value, 4.0);
        panel.observe(&HashMap::from([(id, 4.0)]), false);
        assert_eq!((panel.list[0].value, panel.list[0].pending), (4.0, None));
        assert_eq!(panel.tuned(), [("a.speed".to_owned(), 4.0)]);
        assert!(panel.listing().starts_with("*a.speed  4  (default 1, 0 to 4)"));
        // A load the game took back: the state's word.
        panel.observe(&HashMap::new(), false);
        assert_eq!(panel.list[0].value, 1.0);
        assert!(panel.tuned().is_empty());
        // Paused, a value sent waits for the game to run.
        panel.set(1, 2.5);
        if let Some((_, at)) = &mut panel.list[1].pending {
            *at -= UNSEEN * 2;
        }
        panel.observe(&HashMap::new(), true);
        assert_eq!(panel.list[1].value, 2.5);
        panel.observe(&HashMap::new(), false);
        assert_eq!(panel.list[1].value, 2.5, "the game runs again: it has a while to show it");
        assert_eq!(panel.index("b.wait"), Some(1));
        // It lays out (without a window): a row each, nothing asked for.
        let context = egui::Context::default();
        let mut actions = Vec::new();
        let mut output = context.run_ui(egui::RawInput::default(), |ui| panel.show(ui, &mut actions));
        output.textures_delta.clear();
        assert!(actions.is_empty());
    }

    #[test]
    fn the_tunables_panel_reads_the_file_again_when_it_changes() {
        let root = std::env::temp_dir().join(format!("pen-tunables-{}", std::process::id()));
        std::fs::create_dir_all(root.join("src")).unwrap();
        let file = tunables::path(&root);
        let declare = |default: f32| {
            let text = format!(r#"{{ "tunables": [{{ "name": "a.b", "default": {default}, "min": 0, "max": 9 }}] }}"#);
            std::fs::write(&file, text).unwrap();
        };
        declare(1.0);
        let mut panel = Tunables::load(Some(&root));
        assert_eq!(panel.list[0].declared.default, 1.0);
        assert!(panel.watch.is_some());
        declare(2.5);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !panel.watch.as_mut().unwrap().changed() {
            assert!(Instant::now() < deadline, "the watch never saw the change");
            std::thread::sleep(Duration::from_millis(10));
        }
        panel.reload();
        assert_eq!((panel.list[0].declared.default, panel.list[0].value), (2.5, 1.0), "the value stays the game's");
        std::fs::write(&file, "{ not json").unwrap();
        panel.reload();
        assert!(panel.problem.is_some() && panel.list.len() == 1, "a broken file keeps the list");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_reload_panel_follows_builds_and_reloads() {
        let mut reload = Reload::new(true);
        reload.take(&line(Source::Build, false, "blot: Game compiled (1.05 s)"));
        assert!(matches!(&reload.build, Build::Compiled(status) if status == "Game compiled (1.05 s)"));
        // A failed build: its status, then its diagnostics, until the next.
        reload.take(&line(Source::Build, true, "blot: Game build failed: 1 error (0.90 s)"));
        reload.take(&line(Source::Build, true, "src/main.blot:3:7: unknown name"));
        assert!(reload.failing());
        assert_eq!(reload.errors, ["src/main.blot:3:7: unknown name"]);
        reload.take(&line(Source::Build, false, "blot: Game compiled (1.10 s)"));
        assert!(!reload.failing() && reload.errors.is_empty());
        // The running build's fate, not the hints that follow it.
        reload.take(&line(Source::Reload, false, "Blot reloaded; game state preserved"));
        reload.take(&line(Source::Reload, false, "dev: Ctrl+Shift+R replays the last 5 s on the new build"));
        assert_eq!(
            reload.runtime.as_ref().map(|(_, text, _)| text.as_str()),
            Some("Blot reloaded; game state preserved")
        );
        reload.take(&line(Source::Game, false, "loaded the saved scene"));
        reload.take(&line(Source::Asset, true, "dev: rock.wgsl failed: no"));
        assert_eq!(reload.events.len(), 6);
    }
}
