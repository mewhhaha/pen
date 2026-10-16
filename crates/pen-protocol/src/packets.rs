// Generated from protocol/schema.json by `just sync-protocol`:
// edit the schema and run that, not this file.
//! The packets crossing between the host and a game: their types,
//! constants, the render packet decoder and the input packet encoder.

/// The packets' meaning, as hosts see it. Asset references are indices into the
/// [`Catalog`]; the decoder has checked each one is in range.
pub mod types {
    /// A column-major 4x4 matrix, as WGSL and the packet lay it out.
    pub type Matrix = [f32; 16];
    /// What an interface node is; hosts expose it with the matching role. A textbox's value
    /// text is the text in it; a log is a list of messages whose newest lines a screen reader
    /// may announce.
    pub mod ui_role {
        pub const GROUP: u32 = 0;
        pub const TEXT: u32 = 1;
        pub const HEADING: u32 = 2;
        pub const BUTTON: u32 = 3;
        pub const CHECKBOX: u32 = 4;
        pub const SWITCH: u32 = 5;
        pub const SLIDER: u32 = 6;
        pub const SPINBUTTON: u32 = 7;
        pub const LIST: u32 = 8;
        pub const LISTITEM: u32 = 9;
        pub const DIALOG: u32 = 10;
        pub const MENU: u32 = 11;
        pub const MENUITEM: u32 = 12;
        pub const TABLIST: u32 = 13;
        pub const TAB: u32 = 14;
        pub const TABPANEL: u32 = 15;
        pub const TOOLBAR: u32 = 16;
        pub const IMAGE: u32 = 17;
        pub const STATUS: u32 = 18;
        pub const PROGRESSBAR: u32 = 19;
        pub const RADIOGROUP: u32 = 20;
        pub const RADIO: u32 = 21;
        pub const TEXTBOX: u32 = 22;
        pub const LOG: u32 = 23;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 24] = ["group", "text", "heading", "button", "checkbox", "switch", "slider", "spinbutton", "list", "listitem", "dialog", "menu", "menuitem", "tablist", "tab", "tabpanel", "toolbar", "image", "status", "progressbar", "radiogroup", "radio", "textbox", "log"];
    }
    /// What assistive technology, or the game's keyboard focus, does to an interface node:
    /// moves the focus to it or away from it, does what clicking it does, steps its value up or
    /// down or sets it (to the UiAction event's value), opens or closes it. A node lists the
    /// actions it takes in its `actions` bits, 1 << code each.
    pub mod ui_action {
        pub const FOCUS: u32 = 0;
        pub const BLUR: u32 = 1;
        pub const ACTIVATE: u32 = 2;
        pub const INCREMENT: u32 = 3;
        pub const DECREMENT: u32 = 4;
        pub const SET_VALUE: u32 = 5;
        pub const EXPAND: u32 = 6;
        pub const COLLAPSE: u32 = 7;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 8] = ["focus", "blur", "activate", "increment", "decrement", "set_value", "expand", "collapse"];
    }
    /// An interface node's checked, expanded or selected state: none for a node without one.
    pub mod ui_state {
        pub const NONE: u32 = 0;
        pub const FALSE: u32 = 1;
        pub const TRUE: u32 = 2;
        pub const MIXED: u32 = 3;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 4] = ["none", "false", "true", "mixed"];
    }
    /// Whether assistive technology announces an interface node's changes as they come, and how
    /// urgently.
    pub mod ui_live {
        pub const OFF: u32 = 0;
        pub const POLITE: u32 = 1;
        pub const ASSERTIVE: u32 = 2;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 3] = ["off", "polite", "assertive"];
    }
    /// Which service a request is for: Steam's achievements, stats and rich presence, the
    /// system clipboard, the browser for a web address, what the game runs on, a native compute
    /// library (grid pathfinding), and the host's developer tools (DEVTOOLS_OP_*), which only a
    /// dev tools program asks of. A host without one answers unavailable.
    pub mod service {
        pub const STEAM: u32 = 0;
        pub const CLIPBOARD: u32 = 1;
        pub const URL: u32 = 2;
        pub const PLATFORM: u32 = 3;
        pub const PATHFINDING: u32 = 4;
        pub const DEVTOOLS: u32 = 5;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 6] = ["steam", "clipboard", "url", "platform", "pathfinding", "devtools"];
    }
    /// How a service request went: done; the service is not there (Steam is off, no clipboard,
    /// the browser cannot); it is there but refused or failed; or the host does not know the
    /// service or operation.
    pub mod service_status {
        pub const OK: u32 = 0;
        pub const UNAVAILABLE: u32 = 1;
        pub const FAILED: u32 = 2;
        pub const UNKNOWN: u32 = 3;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 4] = ["ok", "unavailable", "failed", "unknown"];
    }
    /// Steam's operations. Achievements and stats go by their API names (the text); set_stat
    /// and set_stat_float set `a`, add_stat adds `a` to an integer stat, and these and get_stat
    /// answer the stat's value. set_presence sets rich presence key (the text's first `a` code
    /// points) to the rest of the text; clear_presence clears every key. status answers ok
    /// while Steam runs for the game, unavailable otherwise.
    pub mod steam_op {
        pub const UNLOCK_ACHIEVEMENT: u32 = 0;
        pub const CLEAR_ACHIEVEMENT: u32 = 1;
        pub const SET_STAT: u32 = 2;
        pub const SET_STAT_FLOAT: u32 = 3;
        pub const ADD_STAT: u32 = 4;
        pub const GET_STAT: u32 = 5;
        pub const SET_PRESENCE: u32 = 6;
        pub const CLEAR_PRESENCE: u32 = 7;
        pub const STATUS: u32 = 8;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 9] = ["unlock_achievement", "clear_achievement", "set_stat", "set_stat_float", "add_stat", "get_stat", "set_presence", "clear_presence", "status"];
    }
    /// The clipboard's operations: copy the text to it, or answer with the text on it.
    pub mod clipboard_op {
        pub const COPY: u32 = 0;
        pub const PASTE: u32 = 1;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 2] = ["copy", "paste"];
    }
    /// Open the http or https address in the text in the player's browser.
    pub mod url_op {
        pub const OPEN: u32 = 0;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 1] = ["open"];
    }
    /// What the game runs on, answered as text: the operating system (linux, windows, macos,
    /// web), the player's locale (a BCP 47 tag, en-US), and the device (value 1 on a Steam
    /// Deck, with its name as text).
    pub mod platform_op {
        pub const OS: u32 = 0;
        pub const LOCALE: u32 = 1;
        pub const DEVICE: u32 = 2;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 3] = ["os", "locale", "device"];
    }
    /// The shortest 4-connected path over a grid of `a` columns and `b` rows whose cells are
    /// the request's values (0 open, anything else blocked), from cell `c` to cell `d` (row *
    /// columns + column): it answers the path's cells in order, start and goal included, its
    /// length as the value, or none when there is no path.
    pub mod pathfinding_op {
        pub const FIND: u32 = 0;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 1] = ["find"];
    }
    /// The host's developer tools. log answers the log's lines from serial `a` on, value the
    /// serial after the last, each line as its source (0 game, 1 build, 2 reload, 3 asset, 4
    /// pipeline, 5 session), 1 if an error, its length and its code points. frame answers the
    /// last frame's numbers in DEVTOOLS_FRAME's order, timing the game's frames while `a` is 1.
    /// gpu answers the graphics adapter's name. command sends the game console command `a` with
    /// argument `b`. pause and play stop and start the game's clock; step runs `a` frames and
    /// stays paused; rewind goes back `a` seconds of game time, paused; replay plays the last
    /// `a` seconds again, then live; history keeps the past to travel in (unavailable answers
    /// what needs it while none is kept). record starts recording the input to the text's path,
    /// or a new file when it is empty, or stops a recording. shot saves the window with the dev
    /// tools over it to the text's path, or a new file. tune sets the live-tunable constant
    /// whose id is `a` to `b`, and tuned answers the game's tuned values as id and value pairs.
    /// bake writes them into the checkout as the defaults.
    pub mod devtools_op {
        pub const LOG: u32 = 0;
        pub const FRAME: u32 = 1;
        pub const GPU: u32 = 2;
        pub const COMMAND: u32 = 3;
        pub const PAUSE: u32 = 4;
        pub const PLAY: u32 = 5;
        pub const STEP: u32 = 6;
        pub const REWIND: u32 = 7;
        pub const REPLAY: u32 = 8;
        pub const HISTORY: u32 = 9;
        pub const RECORD: u32 = 10;
        pub const SHOT: u32 = 11;
        pub const TUNE: u32 = 12;
        pub const BAKE: u32 = 13;
        pub const TUNED: u32 = 14;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 15] = ["log", "frame", "gpu", "command", "pause", "play", "step", "rewind", "replay", "history", "record", "shot", "tune", "bake", "tuned"];
    }
    /// Where each number sits in a devtools frame answer: the milliseconds since the frame
    /// before; the game's frame phases in milliseconds (guest, waited for an overlapped guest,
    /// packet decode, validation, render and present, housekeeping, and their total), each -1
    /// when the frame was not timed or did not run; its packet's bytes; the renderer's draw
    /// calls, instanced ones and instances; the frame's meshes and brushes; the game's memory
    /// in bytes; the frame's width and height in pixels; and the timeline: the next frame's
    /// position, 1 when paused, replaying, keeping history or overlapping frames, and the frame
    /// an input recording started at, or -1.
    pub mod devtools_frame {
        pub const INTERVAL: u32 = 0;
        pub const GUEST: u32 = 1;
        pub const WAIT: u32 = 2;
        pub const DECODE: u32 = 3;
        pub const VALIDATE: u32 = 4;
        pub const SUBMIT: u32 = 5;
        pub const HOUSEKEEPING: u32 = 6;
        pub const TOTAL: u32 = 7;
        pub const PACKET: u32 = 8;
        pub const DRAW_CALLS: u32 = 9;
        pub const INSTANCED_DRAW_CALLS: u32 = 10;
        pub const INSTANCES: u32 = 11;
        pub const MESHES: u32 = 12;
        pub const BRUSHES: u32 = 13;
        pub const MEMORY: u32 = 14;
        pub const WIDTH: u32 = 15;
        pub const HEIGHT: u32 = 16;
        pub const POSITION: u32 = 17;
        pub const PAUSED: u32 = 18;
        pub const REPLAYING: u32 = 19;
        pub const HISTORY: u32 = 20;
        pub const PIPELINED: u32 = 21;
        pub const RECORDING: u32 = 22;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 23] = ["interval", "guest", "wait", "decode", "validate", "submit", "housekeeping", "total", "packet", "draw_calls", "instanced_draw_calls", "instances", "meshes", "brushes", "memory", "width", "height", "position", "paused", "replaying", "history", "pipelined", "recording"];
    }
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Vector {
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Color {
        pub r: f32,
        pub g: f32,
        pub b: f32,
        pub a: f32,
    }
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Aberration {
        pub x: f32,
        pub y: f32,
    }
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Drift {
        pub x: f32,
        pub z: f32,
    }
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Viewport {
        pub width: u32,
        pub height: u32,
    }
    #[derive(Clone, Debug, PartialEq)]
    pub enum GameEvent {
        /// A key by its browser `KeyboardEvent.key` name ("a", "A", "ArrowLeft", "Shift", ...);
        /// the game gets its code.
        KeyDown(String),
        KeyUp(String),
        PointerDown { button: u32, x: f32, y: f32 },
        PointerUp { button: u32, x: f32, y: f32 },
        PointerMove { x: f32, y: f32 },
        Wheel { x: f32, y: f32, delta: f32 },
        FocusLost,
        AssetReady { key: String, kind: String, revision: u64 },
        AssetFailed { key: String, kind: String, revision: u64, message: String },
        /// A console command for the game: which, and its argument.
        Console { command: f32, argument: f32 },
        /// A stored preference, played back when the game starts.
        Preference { index: f32, value: f32 },
        /// A live-tunable constant set from the host: which, and its value.
        Tune { index: u32, value: f32 },
        /// An action on a node of the interface the game declared (the render packet's `ui`),
        /// from assistive technology: the node's id, the action (UI_ACTION_*), and the value
        /// set_value sets.
        UiAction { node: u32, action: u32, value: f32 },
        /// The answer to a service request the game made (the render packet's `services`),
        /// once, in a later frame: the request's id, how it went (SERVICE_STATUS_*), a number,
        /// and how many values (a text's code points, or numbers) follow in the ServiceData
        /// events right after it.
        ServiceAnswer { request: u32, status: u32, value: f32, length: u32 },
        /// Three of an answer's values, in order, after its ServiceAnswer: the request's id,
        /// then the values; lanes past the answer's length are zero.
        ServiceData { request: u32, a: f32, b: f32, c: f32 },
    }
    #[derive(Clone, Debug, PartialEq)]
    pub struct Frame {
        /// Seconds on a monotonic clock.
        pub timestamp: f64,
        pub viewport: Viewport,
        pub events: Vec<GameEvent>,
    }
    /// The game's assets, by the IDs the packet uses: each list's index is its ID.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct Catalog {
        pub meshes: Vec<String>,
        pub materials: Vec<String>,
        /// Sound clips, by the IDs the game plays them with.
        pub sounds: Vec<String>,
    }
    impl Catalog {
        pub fn mesh(&self, index: u32) -> &str {
            &self.meshes[index as usize]
        }
        pub fn material(&self, index: u32) -> &str {
            &self.materials[index as usize]
        }
        pub fn sound(&self, index: u32) -> &str {
            &self.sounds[index as usize]
        }
    }
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct Draw {
        pub mesh: u32,
        pub material: u32,
        pub transform: Matrix,
        /// Optional four shader values, overriding the material's instance defaults.
        pub parameters: Option<[f32; 4]>,
        /// Up to 32 more vec4s, flattened, that the shader reads after the parameters; a draw
        /// with any is drawn alone.
        pub extra: Option<Vec<f32>>,
    }
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct SolidFace {
        pub normal: Vector,
        pub distance: f32,
        pub material: u32,
    }
    /// A convex solid: the points behind every face plane (normal · p <= distance).
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct Solid {
        pub faces: Vec<SolidFace>,
    }
    /// `material`'s texture laid onto the surfaces inside the box, the cube -1..1 under
    /// `transform`, along its y axis, where their material takes decals; `opacity` fades it.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Decal {
        pub material: u32,
        pub transform: Matrix,
        pub opacity: f32,
    }
    /// A sound to play once on a bus at the gain, pan, pitch, lowpass cutoff and reverb send
    /// given, all worked out by the game.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Sound {
        pub clip: u32,
        /// The bus it plays through; one the frame lacks passes it on to the master as it is.
        pub bus: u32,
        pub gain: f32,
        pub pan: f32,
        pub pitch: f32,
        pub lowpass: f32,
        /// How much of it goes to the shared reverb.
        pub reverb: f32,
    }
    /// A rectangle over the screen, in pixels from its top left, drawn by its material's sprite
    /// shader, which reads the four `parameters` as it likes.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Sprite {
        pub x: f32,
        pub y: f32,
        pub width: f32,
        pub height: f32,
        pub u0: f32,
        pub v0: f32,
        pub u1: f32,
        pub v1: f32,
        pub color: [f32; 4],
        pub material: u32,
        pub parameters: [f32; 4],
    }
    /// A small 3D view over the screen, in the rectangle at (x, y).
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct Inset {
        pub x: f32,
        pub y: f32,
        pub width: f32,
        pub height: f32,
        pub view: Matrix,
        pub projection: Matrix,
        /// Its own light: ambient, the way its rays travel, and their color.
        pub ambient: Vector,
        pub light_direction: Vector,
        pub light_color: Vector,
        pub draws: Vec<Draw>,
        /// Block size in screen pixels, or zero for a smooth image.
        pub pixel_size: f32,
    }
    /// A patch of a field: a `width` x `height` grid of values the host keeps for `material`,
    /// which every draw of it reads as a texture at binding 3 (red, half-float). The patch's
    /// `columns` x `rows` values land row by row at column `x`, row `z`; the rest of the field
    /// keeps what it had.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct FieldPatch {
        pub material: u32,
        pub width: u32,
        pub height: u32,
        pub x: u32,
        pub z: u32,
        pub columns: u32,
        pub rows: u32,
        pub values: Vec<f32>,
    }
    /// A node of the interface: its id, the same from frame to frame, its parent, what it is,
    /// where it lies on the screen in pixels from its top left (empty for one that is not
    /// drawn), its states, a range's value and bounds, the actions it takes, and its text: its
    /// name, description and value text one after the other, in code points.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct UiNode {
        /// 1 to MAX_UI_ID, from the node's key, so it stays the same.
        pub id: u32,
        /// Its parent's place in the section plus one, or 0 at the top. Parents come first:
        /// hosts put a node whose parent does not come before it at the top.
        pub parent: u32,
        pub role: u32,
        pub x: f32,
        pub y: f32,
        pub width: f32,
        pub height: f32,
        pub focusable: bool,
        /// It has the game's keyboard focus.
        pub focused: bool,
        pub disabled: bool,
        /// Assistive technology leaves it out.
        pub hidden: bool,
        /// A dialog that keeps the focus while it is open.
        pub modal: bool,
        pub checked: u32,
        pub expanded: u32,
        pub selected: u32,
        pub live: u32,
        /// A slider's or spin button's value, in its range.
        pub value: f32,
        pub minimum: f32,
        pub maximum: f32,
        pub step: f32,
        pub actions: u32,
        pub text_length: u32,
        pub name_length: u32,
        pub description_length: u32,
        /// Its name's code points, then its description's, then its value text's.
        pub text: Vec<f32>,
    }
    /// A request to a service: its id, unique in the run and never in two packets, which its
    /// answer carries back; the service (SERVICE_*) and its operation; four numbers; and its
    /// values: text as code points (an API name, the text to copy, an address), or the numbers
    /// a compute service takes.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct ServiceRequest {
        /// 1 to MAX_SERVICE_ID.
        pub request: u32,
        pub service: u32,
        pub operation: u32,
        pub a: f32,
        pub b: f32,
        pub c: f32,
        pub d: f32,
        pub value_count: u32,
        /// Its text's code points, or its numbers.
        pub values: Vec<f32>,
    }
    /// A bus: the voices and sounds on it mix through its lowpass and gain into the master, and
    /// `reverb` of that goes to the shared reverb, as do their own sends through its lowpass
    /// and gain. Bus 0 is the master: everything passes through it last, the reverb's return
    /// included.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Bus {
        pub gain: f32,
        pub lowpass: f32,
        /// How much of it goes to the shared reverb; on the master, how loud the reverb comes
        /// back.
        pub reverb: f32,
    }
    /// A voice the game keeps playing, declared by every frame that has it, whole. Hosts start
    /// a new id from `start`, fading in over `fade_in`; fade out an id the frame leaves out
    /// over its `fade_out` (at least AUDIO_FADE_MS), and one that comes back while fading
    /// returns where it is (over its `fade_in`, at least AUDIO_FADE_MS); start one whose clip
    /// or bus changed afresh, the old fading out; and glide the values that change
    /// (AUDIO_GLIDE_MS). A voice that does not loop plays once and is then silent for as long
    /// as frames keep it.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Voice {
        /// 1 to MAX_VOICE_ID, the same from frame to frame while it plays.
        pub id: u32,
        pub clip: u32,
        /// The bus it plays through; one the frame lacks passes it on to the master as it is.
        pub bus: u32,
        pub gain: f32,
        pub pan: f32,
        pub pitch: f32,
        pub lowpass: f32,
        /// How much of it goes to the shared reverb.
        pub reverb: f32,
        pub looping: bool,
        /// Seconds into the clip it starts from, round a looping clip as often as need be.
        pub start: f32,
        /// Seconds it fades in over as it starts.
        pub fade_in: f32,
        /// Seconds it fades out over once frames leave it out.
        pub fade_out: f32,
    }
    /// How the game wants this frame drawn: samples per pixel, FXAA, render scale, texture
    /// filtering, ambient occlusion, a sky, screen effects and bloom.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Quality {
        /// Samples per pixel: 1 or 4.
        pub msaa: u32,
        pub fxaa: bool,
        pub scale: f32,
        /// 0 linear, 1 trilinear, otherwise anisotropic samples.
        pub filtering: u32,
        pub occlusion: bool,
        pub sky: bool,
        pub effects: bool,
        pub bloom: bool,
    }
    /// The sky's colors and the way toward the sun or moon.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Sky {
        pub zenith: Vector,
        pub horizon: Vector,
        pub ground: Vector,
        pub sun: Vector,
        /// 1 at full night, when stars show.
        pub night: f32,
    }
    /// Mist beyond `radius` of the middle; density 0 for none.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Fog {
        pub radius: f32,
        pub density: f32,
        pub height: f32,
    }
    /// The shadow map's view; `texels` 0 for no shadows.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Shadow {
        pub view: Matrix,
        pub projection: Matrix,
        /// A map texel's width in world units.
        pub texel: f32,
        pub texels: u32,
    }
    /// Clouds over the scene: how much of the sky they cover, 0 for none, and how far the wind
    /// has carried them along x and z.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Clouds {
        pub cover: f32,
        pub drift: Drift,
    }
    /// The game's frame: what to draw and play, and what to do.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct RenderFrame {
        pub save_requested: bool,
        pub load_requested: bool,
        pub scene_save_requested: bool,
        pub scene_load_requested: bool,
        /// An edit finished: the host records the scene for undo.
        pub checkpoint_requested: bool,
        pub undo_requested: bool,
        pub redo_requested: bool,
        /// The game is over: the host stops it.
        pub exit_requested: bool,
        pub view: Matrix,
        pub projection: Matrix,
        pub clear: Color,
        pub ambient: Vector,
        pub light_direction: Vector,
        pub light_color: Vector,
        /// How far red moves from green, in screen heights across and down, and blue as far the
        /// other way, on the surfaces whose material asks for it.
        pub aberration: Aberration,
        /// How the game wants this frame drawn: samples per pixel, FXAA, render scale, texture
        /// filtering, ambient occlusion, a sky, screen effects and bloom.
        pub quality: Quality,
        /// The sky's colors and the way toward the sun or moon.
        pub sky: Sky,
        /// Mist beyond `radius` of the middle; density 0 for none.
        pub fog: Fog,
        /// The shadow map's view; `texels` 0 for no shadows.
        pub shadow: Shadow,
        /// How far the picture smears toward the middle of the screen, as a share of the way
        /// there (0 to 0.5), as speed would.
        pub blur: f32,
        /// Clouds over the scene: how much of the sky they cover, 0 for none, and how far the
        /// wind has carried them along x and z.
        pub clouds: Clouds,
        /// The frame supplies its own cursor and asks the host to hide the system one.
        pub hide_cursor: bool,
        pub draws: Vec<Draw>,
        /// Convex solids: the points behind every face plane (normal · p <= distance).
        pub solids: Vec<Solid>,
        /// Decals: `material`'s texture laid onto the surfaces inside the box, the cube -1..1
        /// under `transform`, along its y axis, where their material takes decals; `opacity`
        /// fades it.
        pub decals: Vec<Decal>,
        /// Sounds to play once, each on a bus: gain, pan, pitch, lowpass and reverb send, all
        /// worked out by the game.
        pub sounds: Vec<Sound>,
        pub sprites: Vec<Sprite>,
        pub insets: Vec<Inset>,
        /// The player's preferences to store, when they changed; the host plays them back as
        /// Preference events when the game starts.
        pub preferences: Option<Vec<f32>>,
        /// Patches of the values the host keeps for some materials.
        pub fields: Vec<FieldPatch>,
        /// The interface the game declares for assistive technology and keyboard focus, every
        /// frame, whole: its nodes, parents first.
        pub ui: Vec<UiNode>,
        /// The frame's requests to services, each in one packet only: hosts answer each once,
        /// with ServiceAnswer and ServiceData events in a later frame, and a replay, whose
        /// recorded input holds the answers, asks no service again.
        pub services: Vec<ServiceRequest>,
        /// The mix's buses, every frame, whole: the master first. A frame without buses plays
        /// everything through an open master at gain 1, the reverb coming back at 1; a bus it
        /// lacks is open and dry at gain 1.
        pub buses: Vec<Bus>,
        /// The voices the game keeps playing, every frame, whole: hosts start, fade out and
        /// glide them by id.
        pub voices: Vec<Voice>,
    }
}

/// The packets' layout: constants, the input packet encoder and the render
/// packet decoder.
pub mod codec {
    use anyhow::{Result, bail, ensure};

    use super::types::*;

    /// Counts travel in float32 lanes: the wire format's exact-integer range, not a runtime
    /// memory budget.
    pub const MAX_STATE_FLOATS: usize = 16777215;
    pub const MAX_EVENTS: usize = 100000;
    pub const MAX_DRAWS: usize = 100000;
    pub const MAX_SOLIDS: usize = 100000;
    /// Faces of one convex solid; games may use fewer.
    pub const MAX_SOLID_FACES: usize = 64;
    pub const MAX_DECALS: usize = 4096;
    pub const MAX_SOUNDS: usize = 256;
    pub const MAX_SPRITES: usize = 65536;
    pub const MAX_INSETS: usize = 16;
    pub const MAX_FIELD_PATCHES: usize = 4096;
    /// The widest or tallest field.
    pub const MAX_FIELD: usize = 4096;
    /// The most extra vec4s one draw may carry.
    pub const MAX_DRAW_EXTRA: usize = 32;
    /// The most nodes one frame's interface may have.
    pub const MAX_UI_NODES: usize = 4096;
    /// The most code points one interface node's name, description and value text may have
    /// together.
    pub const MAX_UI_TEXT: usize = 1024;
    /// Interface node ids are whole numbers from 1 to this, the wire's exact-integer range.
    pub const MAX_UI_ID: usize = 16777215;
    /// The most service requests one frame may make.
    pub const MAX_SERVICE_REQUESTS: usize = 256;
    /// The most values (a text's code points, or a compute service's numbers) one service
    /// request or answer carries.
    pub const MAX_SERVICE_DATA: usize = 16384;
    /// Service request ids are whole numbers from 1 to this, the wire's exact-integer range.
    pub const MAX_SERVICE_ID: usize = 16777215;
    /// Operation codes run from 0 to this; each service numbers its own.
    pub const MAX_SERVICE_OPERATION: usize = 255;
    /// The most voices one frame may declare.
    pub const MAX_VOICES: usize = 256;
    /// The most buses one frame may declare, the master (bus 0) included.
    pub const MAX_BUSES: usize = 16;
    /// Voice ids are whole numbers from 1 to this, the wire's exact-integer range.
    pub const MAX_VOICE_ID: usize = 16777215;
    /// The time constant, in milliseconds, of the glide that takes a voice's or bus's gain,
    /// pan, pitch, lowpass and reverb toward a frame's new value (WebAudio's setTargetAtTime),
    /// so that values set every frame do not click.
    pub const AUDIO_GLIDE_MS: usize = 20;
    /// The shortest fade, in milliseconds, of a voice that leaves or changes clip.
    pub const AUDIO_FADE_MS: usize = 30;
    /// A lowpass cutoff, in Hz, at or above which a filter is open; hosts also leave one open
    /// at or above half their sample rate.
    pub const AUDIO_OPEN: usize = 24000;
    pub const PACKET_MAGIC: f32 = 0x474446 as f32;
    /// The render packet header: magic, schema, state_count, draw_count, solid_count,
    /// solid_lanes, decal_count, sound_count, sprite_count, inset_count, inset_draw_count,
    /// preference_count, save_requested, load_requested, scene_save_requested,
    /// scene_load_requested, checkpoint_requested, undo_requested, redo_requested,
    /// exit_requested, view16, projection16, clear4, ambient3, light_direction3, light_color3,
    /// aberration2, quality.msaa, quality.fxaa, quality.scale, quality.filtering,
    /// quality.occlusion, quality.sky, quality.effects, sky.zenith3, sky.horizon3, sky.ground3,
    /// sky.sun3, sky.night, fog.radius, fog.density, fog.height, shadow.view16,
    /// shadow.projection16, shadow.texel, shadow.texels, blur, quality.bloom, clouds.cover,
    /// clouds.drift2, hide_cursor, field_count, field_lanes, ui_count, ui_lanes, service_count,
    /// service_lanes, bus_count, voice_count.
    pub const PACKET_HEADER_FLOATS: usize = 138;
    /// The input packet header: elapsed, width, height, event count and state count.
    pub const FRAME_HEADER_FLOATS: usize = 5;
    /// An event: its tag, then code, x, y, delta.
    pub const EVENT_FLOATS: usize = 5;
    /// Lanes: mesh, material, transform16, parameters count and parameters4, extra count.
    pub const DRAW_FLOATS: usize = 24;
    /// Lanes: faces count, before a solid's faces.
    pub const SOLID_FLOATS: usize = 1;
    /// Lanes: normal3, distance, material.
    pub const SOLID_FACE_FLOATS: usize = 5;
    /// Lanes: material, transform16, opacity.
    pub const DECAL_FLOATS: usize = 18;
    /// Lanes: clip, bus, gain, pan, pitch, lowpass, reverb.
    pub const SOUND_FLOATS: usize = 7;
    /// Lanes: x, y, width, height, u0, v0, u1, v1, color4, material, parameters4.
    pub const SPRITE_FLOATS: usize = 17;
    /// Lanes: x, y, width, height, view16, projection16, ambient3, light_direction3,
    /// light_color3, draws count, pixel_size.
    pub const INSET_FLOATS: usize = 47;
    /// Lanes: preferences lanes.
    pub const PREFERENCE_FLOATS: usize = 14;
    /// Lanes: material, width, height, x, z, columns, rows, before a field's values.
    pub const FIELD_FLOATS: usize = 7;
    /// Lanes: id, parent, role, x, y, width, height, focusable, focused, disabled, hidden,
    /// modal, checked, expanded, selected, live, value, minimum, maximum, step, actions,
    /// text_length, name_length, description_length, before a ui's text.
    pub const UI_NODE_FLOATS: usize = 24;
    /// Lanes: request, service, operation, a, b, c, d, value_count, before a service's values.
    pub const SERVICE_REQUEST_FLOATS: usize = 8;
    /// Lanes: gain, lowpass, reverb.
    pub const BUS_FLOATS: usize = 3;
    /// Lanes: id, clip, bus, gain, pan, pitch, lowpass, reverb, looping, start, fade_in,
    /// fade_out.
    pub const VOICE_FLOATS: usize = 12;
    /// Where each header lane and record field starts; an optional field's lanes
    /// follow its count.
    pub mod lanes {
        pub mod header {
            pub const MAGIC: usize = 0;
            pub const SCHEMA: usize = 1;
            pub const STATE_COUNT: usize = 2;
            pub const DRAW_COUNT: usize = 3;
            pub const SOLID_COUNT: usize = 4;
            pub const SOLID_LANES: usize = 5;
            pub const DECAL_COUNT: usize = 6;
            pub const SOUND_COUNT: usize = 7;
            pub const SPRITE_COUNT: usize = 8;
            pub const INSET_COUNT: usize = 9;
            pub const INSET_DRAW_COUNT: usize = 10;
            pub const PREFERENCE_COUNT: usize = 11;
            pub const SAVE_REQUESTED: usize = 12;
            pub const LOAD_REQUESTED: usize = 13;
            pub const SCENE_SAVE_REQUESTED: usize = 14;
            pub const SCENE_LOAD_REQUESTED: usize = 15;
            pub const CHECKPOINT_REQUESTED: usize = 16;
            pub const UNDO_REQUESTED: usize = 17;
            pub const REDO_REQUESTED: usize = 18;
            pub const EXIT_REQUESTED: usize = 19;
            pub const VIEW: usize = 20;
            pub const PROJECTION: usize = 36;
            pub const CLEAR: usize = 52;
            pub const AMBIENT: usize = 56;
            pub const LIGHT_DIRECTION: usize = 59;
            pub const LIGHT_COLOR: usize = 62;
            pub const ABERRATION: usize = 65;
            pub const QUALITY_MSAA: usize = 67;
            pub const QUALITY_FXAA: usize = 68;
            pub const QUALITY_SCALE: usize = 69;
            pub const QUALITY_FILTERING: usize = 70;
            pub const QUALITY_OCCLUSION: usize = 71;
            pub const QUALITY_SKY: usize = 72;
            pub const QUALITY_EFFECTS: usize = 73;
            pub const SKY_ZENITH: usize = 74;
            pub const SKY_HORIZON: usize = 77;
            pub const SKY_GROUND: usize = 80;
            pub const SKY_SUN: usize = 83;
            pub const SKY_NIGHT: usize = 86;
            pub const FOG_RADIUS: usize = 87;
            pub const FOG_DENSITY: usize = 88;
            pub const FOG_HEIGHT: usize = 89;
            pub const SHADOW_VIEW: usize = 90;
            pub const SHADOW_PROJECTION: usize = 106;
            pub const SHADOW_TEXEL: usize = 122;
            pub const SHADOW_TEXELS: usize = 123;
            pub const BLUR: usize = 124;
            pub const QUALITY_BLOOM: usize = 125;
            pub const CLOUDS_COVER: usize = 126;
            pub const CLOUDS_DRIFT: usize = 127;
            pub const HIDE_CURSOR: usize = 129;
            pub const FIELD_COUNT: usize = 130;
            pub const FIELD_LANES: usize = 131;
            pub const UI_COUNT: usize = 132;
            pub const UI_LANES: usize = 133;
            pub const SERVICE_COUNT: usize = 134;
            pub const SERVICE_LANES: usize = 135;
            pub const BUS_COUNT: usize = 136;
            pub const VOICE_COUNT: usize = 137;
        }
        pub mod draw {
            pub const MESH: usize = 0;
            pub const MATERIAL: usize = 1;
            pub const TRANSFORM: usize = 2;
            pub const PARAMETERS: usize = 18;
            pub const EXTRA: usize = 23;
        }
        pub mod solid_face {
            pub const NORMAL: usize = 0;
            pub const DISTANCE: usize = 3;
            pub const MATERIAL: usize = 4;
        }
        pub mod decal {
            pub const MATERIAL: usize = 0;
            pub const TRANSFORM: usize = 1;
            pub const OPACITY: usize = 17;
        }
        pub mod sound {
            pub const CLIP: usize = 0;
            pub const BUS: usize = 1;
            pub const GAIN: usize = 2;
            pub const PAN: usize = 3;
            pub const PITCH: usize = 4;
            pub const LOWPASS: usize = 5;
            pub const REVERB: usize = 6;
        }
        pub mod sprite {
            pub const X: usize = 0;
            pub const Y: usize = 1;
            pub const WIDTH: usize = 2;
            pub const HEIGHT: usize = 3;
            pub const U0: usize = 4;
            pub const V0: usize = 5;
            pub const U1: usize = 6;
            pub const V1: usize = 7;
            pub const COLOR: usize = 8;
            pub const MATERIAL: usize = 12;
            pub const PARAMETERS: usize = 13;
        }
        pub mod inset {
            pub const X: usize = 0;
            pub const Y: usize = 1;
            pub const WIDTH: usize = 2;
            pub const HEIGHT: usize = 3;
            pub const VIEW: usize = 4;
            pub const PROJECTION: usize = 20;
            pub const AMBIENT: usize = 36;
            pub const LIGHT_DIRECTION: usize = 39;
            pub const LIGHT_COLOR: usize = 42;
            pub const DRAWS: usize = 45;
            pub const PIXEL_SIZE: usize = 46;
        }
        pub mod bus {
            pub const GAIN: usize = 0;
            pub const LOWPASS: usize = 1;
            pub const REVERB: usize = 2;
        }
        pub mod voice {
            pub const ID: usize = 0;
            pub const CLIP: usize = 1;
            pub const BUS: usize = 2;
            pub const GAIN: usize = 3;
            pub const PAN: usize = 4;
            pub const PITCH: usize = 5;
            pub const LOWPASS: usize = 6;
            pub const REVERB: usize = 7;
            pub const LOOPING: usize = 8;
            pub const START: usize = 9;
            pub const FADE_IN: usize = 10;
            pub const FADE_OUT: usize = 11;
        }
        pub mod solid {
            pub const FACES: usize = 0;
        }
        pub mod field_patch {
            pub const MATERIAL: usize = 0;
            pub const WIDTH: usize = 1;
            pub const HEIGHT: usize = 2;
            pub const X: usize = 3;
            pub const Z: usize = 4;
            pub const COLUMNS: usize = 5;
            pub const ROWS: usize = 6;
        }
        pub mod ui_node {
            pub const ID: usize = 0;
            pub const PARENT: usize = 1;
            pub const ROLE: usize = 2;
            pub const X: usize = 3;
            pub const Y: usize = 4;
            pub const WIDTH: usize = 5;
            pub const HEIGHT: usize = 6;
            pub const FOCUSABLE: usize = 7;
            pub const FOCUSED: usize = 8;
            pub const DISABLED: usize = 9;
            pub const HIDDEN: usize = 10;
            pub const MODAL: usize = 11;
            pub const CHECKED: usize = 12;
            pub const EXPANDED: usize = 13;
            pub const SELECTED: usize = 14;
            pub const LIVE: usize = 15;
            pub const VALUE: usize = 16;
            pub const MINIMUM: usize = 17;
            pub const MAXIMUM: usize = 18;
            pub const STEP: usize = 19;
            pub const ACTIONS: usize = 20;
            pub const TEXT_LENGTH: usize = 21;
            pub const NAME_LENGTH: usize = 22;
            pub const DESCRIPTION_LENGTH: usize = 23;
        }
        pub mod service_request {
            pub const REQUEST: usize = 0;
            pub const SERVICE: usize = 1;
            pub const OPERATION: usize = 2;
            pub const A: usize = 3;
            pub const B: usize = 4;
            pub const C: usize = 5;
            pub const D: usize = 6;
            pub const VALUE_COUNT: usize = 7;
        }
    }
    /// Each event's tag.
    pub mod event_tags {
        pub const KEY_DOWN: u32 = 1;
        pub const KEY_UP: u32 = 2;
        pub const POINTER_DOWN: u32 = 3;
        pub const POINTER_UP: u32 = 4;
        pub const POINTER_MOVE: u32 = 5;
        pub const WHEEL: u32 = 6;
        pub const FOCUS_LOST: u32 = 7;
        pub const ASSET_CHANGED: u32 = 8;
        pub const CONSOLE: u32 = 9;
        pub const PREFERENCE: u32 = 10;
        pub const TUNE: u32 = 11;
        pub const UI_ACTION: u32 = 12;
        pub const SERVICE_ANSWER: u32 = 13;
        pub const SERVICE_DATA: u32 = 14;
    }

    pub fn validate_state(state: &[f32]) -> Result<()> {
        ensure!(
            state.len() <= MAX_STATE_FLOATS && state.iter().all(|value| value.is_finite()),
            "game state must be a bounded array of finite float32 values"
        );
        Ok(())
    }

    pub fn validate_schema(schema: f32) -> Result<u32> {
        ensure!(
            schema.fract() == 0.0 && (1.0..=MAX_STATE_FLOATS as f32).contains(&schema),
            "state schema must be an integer in 1..16777215"
        );
        Ok(schema as u32)
    }

    /// The code a key travels as: named keys by their browser names, other single
    /// UTF-16 code units as themselves (as `charCodeAt(0)` reads them), lowercase
    /// ASCII letters as their capitals, and anything else as 0.
    pub fn key_code(key: &str) -> f32 {
        let named = match key {
            "ArrowLeft" => 0x110100,
            "ArrowRight" => 0x110101,
            "ArrowUp" => 0x110102,
            "ArrowDown" => 0x110103,
            "Tab" => 0x110104,
            "Delete" => 0x110105,
            "Escape" => 0x110106,
            "Backspace" => 0x110107,
            "Enter" => 0x110108,
            "PageUp" => 0x110109,
            "PageDown" => 0x11010a,
            "Home" => 0x11010b,
            "End" => 0x11010c,
            "F2" => 0x110202,
            "F6" => 0x110206,
            "F9" => 0x110209,
            "F10" => 0x11020a,
            "Shift" => 0x110300,
            "Control" => 0x110301,
            "Alt" => 0x110302,
            "Meta" => 0x110303,
            _ => 0,
        };
        if named != 0 {
            return named as f32;
        }
        let mut units = key.encode_utf16();
        match (units.next(), units.next()) {
            (Some(code), None) => {
                if (97..=122).contains(&code) {
                    (code - 32) as f32
                } else {
                    code as f32
                }
            }
            _ => 0.0,
        }
    }

    fn event_record(event: &GameEvent) -> [f32; 5] {
        match event {
            GameEvent::KeyDown(value) => [1.0, key_code(value), 0.0, 0.0, 0.0],
            GameEvent::KeyUp(value) => [2.0, key_code(value), 0.0, 0.0, 0.0],
            GameEvent::PointerDown { button, x, y } => [3.0, *button as f32, *x, *y, 0.0],
            GameEvent::PointerUp { button, x, y } => [4.0, *button as f32, *x, *y, 0.0],
            GameEvent::PointerMove { x, y } => [5.0, 0.0, *x, *y, 0.0],
            GameEvent::Wheel { x, y, delta } => [6.0, 0.0, *x, *y, *delta],
            GameEvent::FocusLost => [7.0, 0.0, 0.0, 0.0, 0.0],
            GameEvent::AssetReady { .. } => [8.0, 0.0, 0.0, 0.0, 0.0],
            GameEvent::AssetFailed { .. } => [8.0, 0.0, 0.0, 0.0, 0.0],
            GameEvent::Console { command, argument } => [9.0, *command, *argument, 0.0, 0.0],
            GameEvent::Preference { index, value } => [10.0, *index, *value, 0.0, 0.0],
            GameEvent::Tune { index, value } => [11.0, *index as f32, *value, 0.0, 0.0],
            GameEvent::UiAction { node, action, value } => [12.0, *action as f32, *node as f32, *value, 0.0],
            GameEvent::ServiceAnswer { request, status, value, length } => [13.0, *status as f32, *request as f32, *value, *length as f32],
            GameEvent::ServiceData { request, a, b, c } => [14.0, *request as f32, *a, *b, *c],
        }
    }

    /// The host's frame: the seconds since the last one, the viewport, its events and the
    /// state. Unused event lanes are zero.
    pub fn encode_frame(frame: &Frame, elapsed: f64, state: &[f32]) -> Result<Vec<f32>> {
        validate_state(state)?;
        ensure!(
            frame.timestamp.is_finite()
                && elapsed.is_finite()
                && elapsed >= 0.0
                && (1..=0xffffff).contains(&frame.viewport.width)
                && (1..=0xffffff).contains(&frame.viewport.height)
                && frame.events.len() <= MAX_EVENTS,
            "invalid frame clock, viewport or event count"
        );
        let mut packet = Vec::with_capacity(FRAME_HEADER_FLOATS + frame.events.len() * EVENT_FLOATS + state.len());
        packet.extend_from_slice(&[
            elapsed as f32,
            frame.viewport.width as f32,
            frame.viewport.height as f32,
            frame.events.len() as f32,
            state.len() as f32,
        ]);
        for event in &frame.events {
            packet.extend_from_slice(&event_record(event));
        }
        packet.extend_from_slice(state);
        ensure!(packet.iter().all(|value| value.is_finite()), "frame contains a nonfinite float32 value");
        Ok(packet)
    }

    /// A decoded render packet and the state lanes it carried.
    pub struct Decoded {
        pub state: Vec<f32>,
        pub render: RenderFrame,
    }

    /// The packet, the catalog, and where the next owned record or tail starts.
    struct Reader<'a> {
        packet: &'a [f32],
        catalog: &'a Catalog,
        inset_draws_at: usize,
        inset_draws_end: usize,
        extras_at: usize,
    }

    impl Reader<'_> {
        #[inline]
        fn count(&self, index: usize, maximum: usize) -> Result<usize> {
            let value = self.packet[index];
            if value.fract() != 0.0 || value < 0.0 || value > maximum as f32 {
                bail!("game returned an invalid frame count or flag");
            }
            Ok(value as usize)
        }

        #[inline]
        fn flag(&self, index: usize) -> Result<bool> {
            Ok(self.count(index, 1)? == 1)
        }

        #[inline]
        fn matrix(&self, index: usize) -> Matrix {
            self.packet[index..index + 16].try_into().unwrap()
        }

        #[inline]
        fn mesh_at(&self, index: usize) -> Result<u32> {
            Ok(self.count(index, self.catalog.meshes.len().saturating_sub(1))? as u32)
        }

        #[inline]
        fn material_at(&self, index: usize) -> Result<u32> {
            Ok(self.count(index, self.catalog.materials.len().saturating_sub(1))? as u32)
        }

        #[inline]
        fn sound_at(&self, index: usize) -> Result<u32> {
            ensure!(!self.catalog.sounds.is_empty(), "game played a sound with no sounds in the catalog");
            Ok(self.count(index, self.catalog.sounds.len() - 1)? as u32)
        }

        fn decode_draw(&mut self, lane: usize, draw: &mut Draw) -> Result<()> {
            let mesh = self.mesh_at(lane)?;
            let material = self.material_at(lane + 1)?;
            let transform = self.matrix(lane + 2);
            let parameters_count = self.count(lane + 18, 4)?;
            ensure!(parameters_count == 0 || parameters_count == 4, "draw parameters must contain zero or four lanes");
            let parameters = (parameters_count == 4).then(|| self.packet[lane + 19..lane + 23].try_into().unwrap());
            // Counted with the packet's length.
            let extra_lanes = self.packet[lane + 23] as usize * 4;
            let extra_first = self.extras_at;
            self.extras_at += extra_lanes;
            let extra = copy_optional(std::mem::take(&mut draw.extra), &self.packet[extra_first..extra_first + extra_lanes]);
            *draw = Draw { mesh, material, transform, parameters, extra };
            Ok(())
        }

        fn decode_solid_face(&mut self, lane: usize, solid_face: &mut SolidFace) -> Result<()> {
            let normal = Vector { x: self.packet[lane], y: self.packet[lane + 1], z: self.packet[lane + 2] };
            let distance = self.packet[lane + 3];
            let material = self.material_at(lane + 4)?;
            *solid_face = SolidFace { normal, distance, material };
            Ok(())
        }

        fn decode_decal(&mut self, lane: usize, decal: &mut Decal) -> Result<()> {
            let material = self.material_at(lane)?;
            let transform = self.matrix(lane + 1);
            let opacity = self.packet[lane + 17];
            *decal = Decal { material, transform, opacity };
            Ok(())
        }

        fn decode_sound(&mut self, lane: usize, sound: &mut Sound) -> Result<()> {
            let clip = self.sound_at(lane)?;
            let bus = self.count(lane + 1, 15)?;
            let gain = self.packet[lane + 2];
            let pan = self.packet[lane + 3];
            let pitch = self.packet[lane + 4];
            let lowpass = self.packet[lane + 5];
            let reverb = self.packet[lane + 6];
            *sound = Sound { clip, bus: bus as u32, gain, pan, pitch, lowpass, reverb };
            Ok(())
        }

        fn decode_sprite(&mut self, lane: usize, sprite: &mut Sprite) -> Result<()> {
            let x = self.packet[lane];
            let y = self.packet[lane + 1];
            let width = self.packet[lane + 2];
            let height = self.packet[lane + 3];
            let u0 = self.packet[lane + 4];
            let v0 = self.packet[lane + 5];
            let u1 = self.packet[lane + 6];
            let v1 = self.packet[lane + 7];
            let color = self.packet[lane + 8..lane + 12].try_into().unwrap();
            let material = self.material_at(lane + 12)?;
            let parameters = self.packet[lane + 13..lane + 17].try_into().unwrap();
            *sprite = Sprite { x, y, width, height, u0, v0, u1, v1, color, material, parameters };
            Ok(())
        }

        fn decode_inset(&mut self, lane: usize, inset: &mut Inset) -> Result<()> {
            let x = self.packet[lane];
            let y = self.packet[lane + 1];
            let width = self.packet[lane + 2];
            let height = self.packet[lane + 3];
            let view = self.matrix(lane + 4);
            let projection = self.matrix(lane + 20);
            let ambient = Vector { x: self.packet[lane + 36], y: self.packet[lane + 37], z: self.packet[lane + 38] };
            let light_direction = Vector { x: self.packet[lane + 39], y: self.packet[lane + 40], z: self.packet[lane + 41] };
            let light_color = Vector { x: self.packet[lane + 42], y: self.packet[lane + 43], z: self.packet[lane + 44] };
            let draws_count = self.count(lane + 45, MAX_DRAWS)?;
            let draws_first = self.inset_draws_at;
            self.inset_draws_at += draws_count * DRAW_FLOATS;
            ensure!(self.inset_draws_at <= self.inset_draws_end, "game returned more inset draws than it counted");
            let mut draws = std::mem::take(&mut inset.draws);
            resize(&mut draws, draws_count);
            for (item, draw) in draws.iter_mut().enumerate() {
                self.decode_draw(draws_first + item * DRAW_FLOATS, draw)?;
            }
            let pixel_size = self.packet[lane + 46];
            *inset = Inset { x, y, width, height, view, projection, ambient, light_direction, light_color, draws, pixel_size };
            Ok(())
        }

        fn decode_bus(&mut self, lane: usize, bus: &mut Bus) -> Result<()> {
            let gain = self.packet[lane];
            let lowpass = self.packet[lane + 1];
            let reverb = self.packet[lane + 2];
            *bus = Bus { gain, lowpass, reverb };
            Ok(())
        }

        fn decode_voice(&mut self, lane: usize, voice: &mut Voice) -> Result<()> {
            let id = self.count(lane, MAX_VOICE_ID)?;
            let clip = self.sound_at(lane + 1)?;
            let bus = self.count(lane + 2, 15)?;
            let gain = self.packet[lane + 3];
            let pan = self.packet[lane + 4];
            let pitch = self.packet[lane + 5];
            let lowpass = self.packet[lane + 6];
            let reverb = self.packet[lane + 7];
            let looping = self.flag(lane + 8)?;
            let start = self.packet[lane + 9];
            let fade_in = self.packet[lane + 10];
            let fade_out = self.packet[lane + 11];
            *voice = Voice { id: id as u32, clip, bus: bus as u32, gain, pan, pitch, lowpass, reverb, looping, start, fade_in, fade_out };
            Ok(())
        }
    }

    /// Decodes a render packet. The game's frame: what to draw and play, and what to do. With
    /// several faults, it reports the first in lane order.
    pub fn decode_frame(packet: &[f32], schema: u32, catalog: &Catalog) -> Result<Decoded> {
        let mut render = RenderFrame::default();
        let state = decode_into(packet, schema, catalog, &mut render)?.to_vec();
        Ok(Decoded { state, render })
    }

    /// Decodes a render packet into `frame` as [`decode_frame`] does, reusing the vectors it
    /// holds, and returns the state lanes the packet carried. After an error, `frame` holds
    /// some of this packet and some of what it held before.
    pub fn decode_into<'a>(
        packet: &'a [f32],
        schema: u32,
        catalog: &Catalog,
        frame: &mut RenderFrame,
    ) -> Result<&'a [f32]> {
        ensure!(
            packet.len() >= PACKET_HEADER_FLOATS && packet[0] == PACKET_MAGIC && packet[1] == schema as f32,
            "game returned an invalid frame header or state schema"
        );
        let mut reader = Reader {
            packet,
            catalog,
            inset_draws_at: 0,
            inset_draws_end: 0,
            extras_at: 0,
        };
        // The sections' sizes.
        let state_lanes = reader.count(2, MAX_STATE_FLOATS)?;
        let draws_count = reader.count(3, MAX_DRAWS)?;
        let solids_count = reader.count(4, MAX_SOLIDS)?;
        let solids_lanes = reader.count(5, 32100000)?;
        let decals_count = reader.count(6, MAX_DECALS)?;
        let sounds_count = reader.count(7, MAX_SOUNDS)?;
        let sprites_count = reader.count(8, MAX_SPRITES)?;
        let insets_count = reader.count(9, MAX_INSETS)?;
        let inset_draws_count = reader.count(10, MAX_DRAWS)?;
        let preferences_lanes = reader.count(11, PREFERENCE_FLOATS)?;
        ensure!(preferences_lanes == 0 || preferences_lanes == 14, "game returned partial preferences");
        let fields_count = reader.count(130, MAX_FIELD_PATCHES)?;
        let fields_lanes = reader.count(131, 16805888)?;
        let ui_count = reader.count(132, MAX_UI_NODES)?;
        let ui_lanes = reader.count(133, 4292608)?;
        let services_count = reader.count(134, MAX_SERVICE_REQUESTS)?;
        let services_lanes = reader.count(135, 4196352)?;
        let buses_count = reader.count(136, MAX_BUSES)?;
        let voices_count = reader.count(137, MAX_VOICES)?;
        // Where the sections lie.
        let state_start = PACKET_HEADER_FLOATS;
        let state_end = state_start + state_lanes;
        let draws_start = state_end;
        let draws_end = draws_start + draws_count * DRAW_FLOATS;
        let solids_start = draws_end;
        let solids_end = solids_start + solids_lanes;
        let decals_start = solids_end;
        let decals_end = decals_start + decals_count * DECAL_FLOATS;
        let sounds_start = decals_end;
        let sounds_end = sounds_start + sounds_count * SOUND_FLOATS;
        let sprites_start = sounds_end;
        let sprites_end = sprites_start + sprites_count * SPRITE_FLOATS;
        let insets_start = sprites_end;
        let insets_end = insets_start + insets_count * INSET_FLOATS;
        let inset_draws_start = insets_end;
        let inset_draws_end = inset_draws_start + inset_draws_count * DRAW_FLOATS;
        let preferences_start = inset_draws_end;
        let preferences_end = preferences_start + preferences_lanes;
        let fields_start = preferences_end;
        let fields_end = fields_start + fields_lanes;
        let ui_start = fields_end;
        let ui_end = ui_start + ui_lanes;
        let services_start = ui_end;
        let services_end = services_start + services_lanes;
        let buses_start = services_end;
        let buses_end = buses_start + buses_count * BUS_FLOATS;
        let voices_start = buses_end;
        let voices_end = voices_start + voices_count * VOICE_FLOATS;
        let extras_start = voices_end;
        let truncated = "game returned a truncated, oversized or nonfinite frame";
        ensure!(packet.len() >= voices_end && packet.iter().all(|value| value.is_finite()), truncated);
        // What the records' tails claim of extras, in order.
        let mut extras_lanes = 0;
        for index in 0..draws_count {
            extras_lanes += reader.count(draws_start + index * DRAW_FLOATS + 23, MAX_DRAW_EXTRA)? * 4;
        }
        for index in 0..inset_draws_count {
            extras_lanes += reader.count(inset_draws_start + index * DRAW_FLOATS + 23, MAX_DRAW_EXTRA)? * 4;
        }
        let extras_end = extras_start + extras_lanes;
        ensure!(packet.len() == extras_end, truncated);
        reader.inset_draws_at = inset_draws_start;
        reader.inset_draws_end = inset_draws_end;
        reader.extras_at = extras_start;
        // The header's values.
        let save_requested = reader.flag(12)?;
        let load_requested = reader.flag(13)?;
        let scene_save_requested = reader.flag(14)?;
        let scene_load_requested = reader.flag(15)?;
        let checkpoint_requested = reader.flag(16)?;
        let undo_requested = reader.flag(17)?;
        let redo_requested = reader.flag(18)?;
        let exit_requested = reader.flag(19)?;
        let view = reader.matrix(20);
        let projection = reader.matrix(36);
        let clear = Color { r: reader.packet[52], g: reader.packet[53], b: reader.packet[54], a: reader.packet[55] };
        let ambient = Vector { x: reader.packet[56], y: reader.packet[57], z: reader.packet[58] };
        let light_direction = Vector { x: reader.packet[59], y: reader.packet[60], z: reader.packet[61] };
        let light_color = Vector { x: reader.packet[62], y: reader.packet[63], z: reader.packet[64] };
        let aberration = Aberration { x: reader.packet[65], y: reader.packet[66] };
        let quality_msaa = if reader.packet[67] == 4.0 { 4 } else { 1 };
        let quality_fxaa = reader.flag(68)?;
        let quality_scale = reader.packet[69];
        let quality_filtering = reader.count(70, 16)?;
        let quality_occlusion = reader.flag(71)?;
        let quality_sky = reader.flag(72)?;
        let quality_effects = reader.flag(73)?;
        let sky_zenith = Vector { x: reader.packet[74], y: reader.packet[75], z: reader.packet[76] };
        let sky_horizon = Vector { x: reader.packet[77], y: reader.packet[78], z: reader.packet[79] };
        let sky_ground = Vector { x: reader.packet[80], y: reader.packet[81], z: reader.packet[82] };
        let sky_sun = Vector { x: reader.packet[83], y: reader.packet[84], z: reader.packet[85] };
        let sky_night = reader.packet[86];
        let fog_radius = reader.packet[87];
        let fog_density = reader.packet[88];
        let fog_height = reader.packet[89];
        let shadow_view = reader.matrix(90);
        let shadow_projection = reader.matrix(106);
        let shadow_texel = reader.packet[122];
        let shadow_texels = reader.count(123, 8192)?;
        let blur = reader.packet[124];
        let quality_bloom = reader.flag(125)?;
        let clouds_cover = reader.packet[126];
        let clouds_drift = Drift { x: reader.packet[127], z: reader.packet[128] };
        let hide_cursor = reader.flag(129)?;
        // The sections.
        let mut draws = std::mem::take(&mut frame.draws);
        resize(&mut draws, draws_count);
        for (index, draw) in draws.iter_mut().enumerate() {
            reader.decode_draw(draws_start + index * DRAW_FLOATS, draw)?;
        }
        let mut solids = std::mem::take(&mut frame.solids);
        resize(&mut solids, solids_count);
        let mut solid_lane = solids_start;
        for solid in solids.iter_mut() {
            ensure!(solid_lane + SOLID_FLOATS <= solids_end, "game returned truncated solids");
            let faces_count = reader.count(solid_lane, MAX_SOLID_FACES)?;
            let faces_first = solid_lane + SOLID_FLOATS;
            let solid_end = faces_first + faces_count * SOLID_FACE_FLOATS;
            ensure!(faces_count >= 4 && solid_end <= solids_end, "game returned an invalid solid");
            let mut faces = std::mem::take(&mut solid.faces);
            resize(&mut faces, faces_count);
            for (item, solid_face) in faces.iter_mut().enumerate() {
                reader.decode_solid_face(faces_first + item * SOLID_FACE_FLOATS, solid_face)?;
            }
            *solid = Solid { faces };
            solid_lane = solid_end;
        }
        ensure!(solid_lane == solids_end, "game returned solid lanes that do not match");
        let mut decals = std::mem::take(&mut frame.decals);
        resize(&mut decals, decals_count);
        for (index, decal) in decals.iter_mut().enumerate() {
            reader.decode_decal(decals_start + index * DECAL_FLOATS, decal)?;
        }
        let mut sounds = std::mem::take(&mut frame.sounds);
        resize(&mut sounds, sounds_count);
        for (index, sound) in sounds.iter_mut().enumerate() {
            reader.decode_sound(sounds_start + index * SOUND_FLOATS, sound)?;
        }
        let mut sprites = std::mem::take(&mut frame.sprites);
        resize(&mut sprites, sprites_count);
        for (index, sprite) in sprites.iter_mut().enumerate() {
            reader.decode_sprite(sprites_start + index * SPRITE_FLOATS, sprite)?;
        }
        let mut insets = std::mem::take(&mut frame.insets);
        resize(&mut insets, insets_count);
        for (index, inset) in insets.iter_mut().enumerate() {
            reader.decode_inset(insets_start + index * INSET_FLOATS, inset)?;
        }
        ensure!(reader.inset_draws_at == inset_draws_end, "game returned inset draws that do not match");
        let preferences = copy_optional(std::mem::take(&mut frame.preferences), &packet[preferences_start..preferences_end]);
        let mut fields = std::mem::take(&mut frame.fields);
        resize(&mut fields, fields_count);
        let mut field_lane = fields_start;
        for field_patch in fields.iter_mut() {
            ensure!(field_lane + FIELD_FLOATS <= fields_end, "game returned truncated field patches");
            let material = reader.material_at(field_lane)?;
            let width = reader.count(field_lane + 1, MAX_FIELD)?;
            let height = reader.count(field_lane + 2, MAX_FIELD)?;
            let x = reader.count(field_lane + 3, width)?;
            let z = reader.count(field_lane + 4, height)?;
            let columns = reader.count(field_lane + 5, width - x)?;
            let rows = reader.count(field_lane + 6, height - z)?;
            let values_first = field_lane + FIELD_FLOATS;
            let field_end = values_first + columns * rows;
            ensure!(field_end <= fields_end, "game returned truncated field patches");
            let values = copy(std::mem::take(&mut field_patch.values), &packet[values_first..field_end]);
            *field_patch = FieldPatch { material, width: width as u32, height: height as u32, x: x as u32, z: z as u32, columns: columns as u32, rows: rows as u32, values };
            field_lane = field_end;
        }
        ensure!(field_lane == fields_end, "game returned field lanes that do not match");
        let mut ui = std::mem::take(&mut frame.ui);
        resize(&mut ui, ui_count);
        let mut ui_record_lane = ui_start;
        for ui_node in ui.iter_mut() {
            ensure!(ui_record_lane + UI_NODE_FLOATS <= ui_end, "game returned truncated interface nodes");
            let id = reader.count(ui_record_lane, MAX_UI_ID)?;
            let parent = reader.count(ui_record_lane + 1, MAX_UI_NODES)?;
            let role = reader.count(ui_record_lane + 2, 23)?;
            let x = reader.packet[ui_record_lane + 3];
            let y = reader.packet[ui_record_lane + 4];
            let width = reader.packet[ui_record_lane + 5];
            let height = reader.packet[ui_record_lane + 6];
            let focusable = reader.flag(ui_record_lane + 7)?;
            let focused = reader.flag(ui_record_lane + 8)?;
            let disabled = reader.flag(ui_record_lane + 9)?;
            let hidden = reader.flag(ui_record_lane + 10)?;
            let modal = reader.flag(ui_record_lane + 11)?;
            let checked = reader.count(ui_record_lane + 12, 3)?;
            let expanded = reader.count(ui_record_lane + 13, 2)?;
            let selected = reader.count(ui_record_lane + 14, 2)?;
            let live = reader.count(ui_record_lane + 15, 2)?;
            let value = reader.packet[ui_record_lane + 16];
            let minimum = reader.packet[ui_record_lane + 17];
            let maximum = reader.packet[ui_record_lane + 18];
            let step = reader.packet[ui_record_lane + 19];
            let actions = reader.count(ui_record_lane + 20, 255)?;
            let text_length = reader.count(ui_record_lane + 21, MAX_UI_TEXT)?;
            let name_length = reader.count(ui_record_lane + 22, text_length)?;
            let description_length = reader.count(ui_record_lane + 23, text_length - name_length)?;
            let text_first = ui_record_lane + UI_NODE_FLOATS;
            let ui_record_end = text_first + text_length;
            ensure!(ui_record_end <= ui_end, "game returned an invalid interface node");
            let text = copy(std::mem::take(&mut ui_node.text), &packet[text_first..ui_record_end]);
            *ui_node = UiNode { id: id as u32, parent: parent as u32, role: role as u32, x, y, width, height, focusable, focused, disabled, hidden, modal, checked: checked as u32, expanded: expanded as u32, selected: selected as u32, live: live as u32, value, minimum, maximum, step, actions: actions as u32, text_length: text_length as u32, name_length: name_length as u32, description_length: description_length as u32, text };
            ui_record_lane = ui_record_end;
        }
        ensure!(ui_record_lane == ui_end, "game returned interface lanes that do not match");
        let mut services = std::mem::take(&mut frame.services);
        resize(&mut services, services_count);
        let mut service_lane = services_start;
        for service_request in services.iter_mut() {
            ensure!(service_lane + SERVICE_REQUEST_FLOATS <= services_end, "game returned truncated service requests");
            let request = reader.count(service_lane, MAX_SERVICE_ID)?;
            let service = reader.count(service_lane + 1, 5)?;
            let operation = reader.count(service_lane + 2, MAX_SERVICE_OPERATION)?;
            let a = reader.packet[service_lane + 3];
            let b = reader.packet[service_lane + 4];
            let c = reader.packet[service_lane + 5];
            let d = reader.packet[service_lane + 6];
            let value_count = reader.count(service_lane + 7, MAX_SERVICE_DATA)?;
            let values_first = service_lane + SERVICE_REQUEST_FLOATS;
            let service_end = values_first + value_count;
            ensure!(service_end <= services_end, "game returned an invalid service request");
            let values = copy(std::mem::take(&mut service_request.values), &packet[values_first..service_end]);
            *service_request = ServiceRequest { request: request as u32, service: service as u32, operation: operation as u32, a, b, c, d, value_count: value_count as u32, values };
            service_lane = service_end;
        }
        ensure!(service_lane == services_end, "game returned service lanes that do not match");
        let mut buses = std::mem::take(&mut frame.buses);
        resize(&mut buses, buses_count);
        for (index, bus) in buses.iter_mut().enumerate() {
            reader.decode_bus(buses_start + index * BUS_FLOATS, bus)?;
        }
        let mut voices = std::mem::take(&mut frame.voices);
        resize(&mut voices, voices_count);
        for (index, voice) in voices.iter_mut().enumerate() {
            reader.decode_voice(voices_start + index * VOICE_FLOATS, voice)?;
        }
        *frame = RenderFrame {
            save_requested,
            load_requested,
            scene_save_requested,
            scene_load_requested,
            checkpoint_requested,
            undo_requested,
            redo_requested,
            exit_requested,
            view,
            projection,
            clear,
            ambient,
            light_direction,
            light_color,
            aberration,
            quality: Quality { msaa: quality_msaa, fxaa: quality_fxaa, scale: quality_scale, filtering: quality_filtering as u32, occlusion: quality_occlusion, sky: quality_sky, effects: quality_effects, bloom: quality_bloom },
            sky: Sky { zenith: sky_zenith, horizon: sky_horizon, ground: sky_ground, sun: sky_sun, night: sky_night },
            fog: Fog { radius: fog_radius, density: fog_density, height: fog_height },
            shadow: Shadow { view: shadow_view, projection: shadow_projection, texel: shadow_texel, texels: shadow_texels as u32 },
            blur,
            clouds: Clouds { cover: clouds_cover, drift: clouds_drift },
            hide_cursor,
            draws,
            solids,
            decals,
            sounds,
            sprites,
            insets,
            preferences,
            fields,
            ui,
            services,
            buses,
            voices,
        };
        Ok(&packet[state_start..state_end])
    }

    /// `records`, `count` long. Those it keeps are decoded over in place, their vectors'
    /// allocations reused.
    fn resize<T: Default>(records: &mut Vec<T>, count: usize) {
        records.truncate(count);
        records.resize_with(count, T::default);
    }

    /// `lanes`, in `values`'s allocation.
    fn copy(mut values: Vec<f32>, lanes: &[f32]) -> Vec<f32> {
        values.clear();
        values.extend_from_slice(lanes);
        values
    }

    /// `lanes`, in `values`'s allocation, or None when there are none.
    fn copy_optional(values: Option<Vec<f32>>, lanes: &[f32]) -> Option<Vec<f32>> {
        (!lanes.is_empty()).then(|| copy(values.unwrap_or_default(), lanes))
    }
}
