// Generated from protocol/schema.json by `just sync-protocol`:
// edit the schema and run that, not this file.
//! The packets crossing between the host and a game: their types,
//! constants, the render packet decoder and the input packet encoder.

/// The packets' meaning, as hosts see it. Asset references are indices into the
/// [`Catalog`]; the decoder has checked each one is in range.
pub mod types {
    /// A column-major 4x4 matrix, as WGSL and the packet lay it out.
    pub type Matrix = [f32; 16];
    /// Version-1 native session test commands. Available only to a separate test guest in pen
    /// test-squid --session.
    pub mod test_driver_op {
        pub const LAUNCH: u32 = 0;
        pub const STEP: u32 = 1;
        pub const CLICK: u32 = 2;
        pub const VISIBLE: u32 = 3;
        pub const HIDDEN: u32 = 4;
        pub const PRESS: u32 = 5;
        pub const SNAPSHOT: u32 = 6;
        pub const EQUAL: u32 = 7;
        pub const SCREENSHOT: u32 = 8;
        pub const FINISH: u32 = 9;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 10] = ["launch", "step", "click", "visible", "hidden", "press", "snapshot", "equal", "screenshot", "finish"];
    }
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
    /// library (grid pathfinding), the host's developer tools (DEVTOOLS_OP_*), which only a dev
    /// tools program asks of, the simulation services both hosts answer alike, bit for bit:
    /// rigid-body physics, paths over a navmesh, ray and shape queries against colliders, and
    /// noise and heightfields, the game's actions and their bindings (INPUT_OP_*), the engine's
    /// preferences (PREFERENCES_OP_*), the player's language (LOCALE_OP_*), the captions of the
    /// sounds that play (CAPTIONS_OP_*), the player's consent to crash and performance reports
    /// (REPORTS_OP_*), and isolated pure data-parallel batch kernels (KERNEL_OP_*). A host
    /// without one answers unavailable.
    pub mod service {
        pub const STEAM: u32 = 0;
        pub const CLIPBOARD: u32 = 1;
        pub const URL: u32 = 2;
        pub const PLATFORM: u32 = 3;
        pub const PATHFINDING: u32 = 4;
        pub const DEVTOOLS: u32 = 5;
        pub const PHYSICS: u32 = 6;
        pub const NAVMESH: u32 = 7;
        pub const SPATIAL: u32 = 8;
        pub const NOISE: u32 = 9;
        pub const INPUT: u32 = 10;
        pub const PREFERENCES: u32 = 11;
        pub const LOCALE: u32 = 12;
        pub const CAPTIONS: u32 = 13;
        pub const REPORTS: u32 = 14;
        pub const KERNEL: u32 = 15;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 16] = ["steam", "clipboard", "url", "platform", "pathfinding", "devtools", "physics", "navmesh", "spatial", "noise", "input", "preferences", "locale", "captions", "reports", "kernel"];
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
    /// Isolated #kernel service operations: ctx_begin, ctx_chunk and ctx_commit upload and
    /// verify a kernel's prepared WireCtx under lossless 32-bit (host_epoch, upload_seq)
    /// ordering, and run_batch executes a batch of items against the bound kernel and context.
    pub mod kernel_op {
        pub const CTX_BEGIN: u32 = 0;
        pub const CTX_CHUNK: u32 = 1;
        pub const CTX_COMMIT: u32 = 2;
        pub const RUN_BATCH: u32 = 3;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 4] = ["ctx_begin", "ctx_chunk", "ctx_commit", "run_batch"];
    }
    /// Steam's operations. Achievements and stats go by their API names (the text); set_stat
    /// and set_stat_float set `a`, add_stat adds `a` to an integer stat, and these and get_stat
    /// answer the stat's value. set_presence sets rich presence key (the text's first `a` code
    /// points) to the rest of the text; clear_presence clears every key. status answers ok
    /// while Steam runs for the game, unavailable otherwise. Steam Input: input_action_set
    /// activates the action set named by the text on every controller, input_glyph answers the
    /// path of the glyph image of the first button or stick bound to the action named by the
    /// text in the active set (a digital action, or an analog one), and input_panel opens
    /// Steam's binding panel, where the player rebinds them; each answers unavailable without
    /// Steam Input or a controller. The Workshop's: workshop_publish uploads the player's saved
    /// scene (the editor's) as an item titled with the text, visible to `a` (0 everyone, 1
    /// friends, 2 only the player, 3 those with the link), or with `b` 1 as a new version of
    /// the item whose id is the text's first line, answered later with the item's id as text
    /// and value 1 when the player must accept the Workshop's agreement; workshop_items answers
    /// the installed subscribed items' ids as text, a space between them, their count the
    /// value, and downloads the rest; workshop_subscribe and workshop_unsubscribe take an
    /// item's id as text and answer once Steam has; workshop_use makes an installed item's
    /// scene the player's, for the game to load as the editor's scene.
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
        pub const INPUT_ACTION_SET: u32 = 9;
        pub const INPUT_GLYPH: u32 = 10;
        pub const INPUT_PANEL: u32 = 11;
        pub const WORKSHOP_PUBLISH: u32 = 12;
        pub const WORKSHOP_ITEMS: u32 = 13;
        pub const WORKSHOP_SUBSCRIBE: u32 = 14;
        pub const WORKSHOP_UNSUBSCRIBE: u32 = 15;
        pub const WORKSHOP_USE: u32 = 16;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 17] = ["unlock_achievement", "clear_achievement", "set_stat", "set_stat_float", "add_stat", "get_stat", "set_presence", "clear_presence", "status", "input_action_set", "input_glyph", "input_panel", "workshop_publish", "workshop_items", "workshop_subscribe", "workshop_unsubscribe", "workshop_use"];
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
    /// bake writes them into the checkout as the defaults, or only the one whose id is `a` when
    /// `a` is not 0. packets keeps each frame's input and render packets from now on while `a`
    /// is 1 (the last 120 frames), or forgets them at 0, and answers the frames kept, oldest
    /// first, each as DEVTOOLS_PACKET's numbers; packet_diff answers how frame `b` differs from
    /// frame `a`, both kept: for each of DEVTOOLS_PACKET's numbers after position, its count in
    /// `a`, its count in `b` and how many of its elements differ. state answers the live
    /// state's sections (`a` below 0), each its key, version, width and length, value the
    /// state's schema, or up to 256 lanes of section `a` from lane `b` on, value its length;
    /// with `c` above 0, of the state the history kept `c` seconds of game time back instead.
    /// state_diff answers, for each section of the live state or the one kept `a` seconds back,
    /// its key, its length then and now, how many lanes differ and the first that does (-1 for
    /// none), value the frame compared with. pictures takes a picture of the game's frame each
    /// time the history could keep a moment, while `a` is 1, into the dev tools' atlas (the
    /// texture of material `b`: slots of 128 by 80 pixels, 8 to a row, the first 16 for
    /// pictures), and answers each picture's slot and the frame that shows it, oldest first;
    /// seek goes to frame `a`, paused. watch pauses the game once lane `b` of the live state's
    /// section `a` compares with `d` as DEVTOOLS_COMPARE `c` says, `a` -1 stops watching and -2
    /// only asks; it answers the watch: key, lane, comparison, value, the lane's value now and
    /// the frame it paused at (-1 for none). profile samples the game's guest while `a` is 1,
    /// stops at 0 and forgets the samples at 2 (3 only asks), answering how many it has and 1
    /// while sampling; flame answers them as a flame graph, value the samples: first how many
    /// functions follow that the samples found running most, each its function index, samples,
    /// its name's length and code points; then each node (a caller before its callees) as its
    /// depth, function index, first sample, samples, samples in it alone, its name's length and
    /// code points. memory answers the game's committed memory in bytes, the most it committed,
    /// then for each counter the guest exports (guest ABI 3's ownership counters) and the
    /// host's calls into it, its name's length, code points and value in two 24-bit halves,
    /// value the number of counters. presets answers the names of the checkout's tunable
    /// presets, a line each; preset_save saves the game's tuned values as the preset the text
    /// names, and preset_load tunes each value of that preset. assets answers the game's
    /// catalog of DEVTOOLS_ASSET `a`, each entry as its DEVTOOLS_ASSET_STATE, its revision, its
    /// key's length and code points. preview draws catalog asset `b` of kind `a` (a mesh with
    /// material `c`, a material on mesh `c`) into slot `d` (16 to 63) of the atlas pictures
    /// named, value 1 once drawn and 0 while its assets load. swap draws catalog asset `b` of
    /// kind `a` from the asset the text names instead, or its own again when the text is empty,
    /// until the next build. material answers material `a`'s color and instance parameters (8
    /// numbers, 0 when it sets none), whether it sets them (two numbers), then its shader's and
    /// texture's names, each its length and code points; material_set sets number `b` of those
    /// 8 in the checkout's file to `c`, which then reloads. audit checks the game's interface
    /// as the last frame showed it and answers its findings: value their count, and for each
    /// its kind (AUDIT_FINDING_*), the node's id, a measure (a contrast ratio, a size in
    /// pixels, or 0), its message's length and the message's code points. audio answers the mix
    /// as it plays: DEVTOOLS_AUDIO's numbers, then DEVTOOLS_BUS's for each bus, the master
    /// first; peaks and RMS are linear and over what played since the last audio answer. mute
    /// silences bus `a` while `b` is 1 and solo hears only the soloed buses (and the master's
    /// own voices) while any is soloed; both are the developer's, never the game's or a
    /// recording's. export hands the game's state over as a save file named by the text, or a
    /// new one (the browser downloads it); import loads a save file the player hands over (the
    /// browser asks for one), or plays an input recording. systems measures named game systems
    /// while a is 1, stops at 0, and answers a bounded page starting at row b (64 rows
    /// maximum). Row zero refreshes the snapshot; subsequent pages read that same snapshot.
    /// Value is the total system count. Each row contains stage (Start through Settle, 0-6),
    /// id, calls since the previous snapshot, mean milliseconds, maximum milliseconds,
    /// execution mode (0 sequential), mean requested allocation bytes per call, mean allocation
    /// count per call, name length and code points. Uncalled systems have -1 timings;
    /// allocation means are -1 without allocation instrumentation or calls. Unsupported hosts
    /// answer unavailable. semantic exchanges bounded JSON text for opt-in application
    /// descriptors, decoded object values, decision events, watches and exact timeline seeks;
    /// see docs/semantic-inspection.md for version 1, limits, pagination and explicit
    /// unavailable reasons.
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
        pub const PACKETS: u32 = 15;
        pub const PACKET_DIFF: u32 = 16;
        pub const STATE: u32 = 17;
        pub const STATE_DIFF: u32 = 18;
        pub const PICTURES: u32 = 19;
        pub const SEEK: u32 = 20;
        pub const WATCH: u32 = 21;
        pub const PROFILE: u32 = 22;
        pub const FLAME: u32 = 23;
        pub const MEMORY: u32 = 24;
        pub const PRESETS: u32 = 25;
        pub const PRESET_SAVE: u32 = 26;
        pub const PRESET_LOAD: u32 = 27;
        pub const ASSETS: u32 = 28;
        pub const PREVIEW: u32 = 29;
        pub const SWAP: u32 = 30;
        pub const MATERIAL: u32 = 31;
        pub const MATERIAL_SET: u32 = 32;
        pub const AUDIT: u32 = 33;
        pub const AUDIO: u32 = 34;
        pub const MUTE: u32 = 35;
        pub const SOLO: u32 = 36;
        pub const EXPORT: u32 = 37;
        pub const IMPORT: u32 = 38;
        pub const SYSTEMS: u32 = 39;
        pub const SEMANTIC: u32 = 40;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 41] = ["log", "frame", "gpu", "command", "pause", "play", "step", "rewind", "replay", "history", "record", "shot", "tune", "bake", "tuned", "packets", "packet_diff", "state", "state_diff", "pictures", "seek", "watch", "profile", "flame", "memory", "presets", "preset_save", "preset_load", "assets", "preview", "swap", "material", "material_set", "audit", "audio", "mute", "solo", "export", "import", "systems", "semantic"];
    }
    /// Where each number sits in a devtools frame answer: the milliseconds since the frame
    /// before; the game's frame phases in milliseconds (guest, waited for an overlapped guest,
    /// packet decode, validation, rendering, surface acquisition, presentation, explicit pacing
    /// waits, housekeeping, and their total), each -1 when the frame was not timed or did not
    /// run. Drawing phases do not overlap: render covers preparation, uploads, command
    /// encoding, queue submission and the developer overlay, excluding acquire and present
    /// calls; acquire and present include any waits inside those calls. Pace covers browser GPU
    /// backpressure and repaint waits (zero natively, whose event-loop pacing is outside the
    /// measured frame). These are wall-clock call durations, not thread CPU time; driver stalls
    /// inside queue submission remain in render. Housekeeping includes audio, input and capture
    /// work. Then its packet's bytes; the renderer's draw calls, instanced ones and instances;
    /// the frame's meshes and brushes; the game's memory in bytes; the frame's width and height
    /// in pixels; and the timeline: the next frame's position, 1 when paused, replaying,
    /// keeping history or overlapping frames, and the frame an input recording started at, or
    /// -1. Then the renderer's culling: draws and instances left out of the camera's view, and
    /// casters left out of the shadow maps (once for each map); the triangles drawn; the render
    /// scale the scene drew at; and the GPU's milliseconds over a recent frame's shadow maps,
    /// culling, scene, the passes after it, sprites and insets, and the whole frame, each -1
    /// where the GPU has no timestamps.
    pub mod devtools_frame {
        pub const INTERVAL: u32 = 0;
        pub const GUEST: u32 = 1;
        pub const WAIT: u32 = 2;
        pub const DECODE: u32 = 3;
        pub const VALIDATE: u32 = 4;
        pub const RENDER: u32 = 5;
        pub const ACQUIRE: u32 = 6;
        pub const PRESENT: u32 = 7;
        pub const PACE: u32 = 8;
        pub const HOUSEKEEPING: u32 = 9;
        pub const TOTAL: u32 = 10;
        pub const PACKET: u32 = 11;
        pub const DRAW_CALLS: u32 = 12;
        pub const INSTANCED_DRAW_CALLS: u32 = 13;
        pub const INSTANCES: u32 = 14;
        pub const MESHES: u32 = 15;
        pub const BRUSHES: u32 = 16;
        pub const MEMORY: u32 = 17;
        pub const WIDTH: u32 = 18;
        pub const HEIGHT: u32 = 19;
        pub const POSITION: u32 = 20;
        pub const PAUSED: u32 = 21;
        pub const REPLAYING: u32 = 22;
        pub const HISTORY: u32 = 23;
        pub const PIPELINED: u32 = 24;
        pub const RECORDING: u32 = 25;
        pub const CULLED: u32 = 26;
        pub const SHADOW_CULLED: u32 = 27;
        pub const TRIANGLES: u32 = 28;
        pub const RENDER_SCALE: u32 = 29;
        pub const GPU_SHADOW: u32 = 30;
        pub const GPU_CULLING: u32 = 31;
        pub const GPU_SCENE: u32 = 32;
        pub const GPU_POST: u32 = 33;
        pub const GPU_OVERLAY: u32 = 34;
        pub const GPU_TOTAL: u32 = 35;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 36] = ["interval", "guest", "wait", "decode", "validate", "render", "acquire", "present", "pace", "housekeeping", "total", "packet", "draw_calls", "instanced_draw_calls", "instances", "meshes", "brushes", "memory", "width", "height", "position", "paused", "replaying", "history", "pipelined", "recording", "culled", "shadow_culled", "triangles", "render_scale", "gpu_shadow", "gpu_culling", "gpu_scene", "gpu_post", "gpu_overlay", "gpu_total"];
    }
    /// Rigid-body physics (rapier), from the world the request sends: step advances it `a`
    /// steps (1 to 600) of `b` seconds (up to 1) with gravity `c` (metres per second squared
    /// along y). The values are its bodies, each a kind (PHYSICS_BODY_*), a collider
    /// (COLLIDER_SHAPE_*'s record: shape, size x y z, position x y z, rotation quaternion x y z
    /// w), linear velocity x y z, angular velocity x y z (radians a second), friction,
    /// restitution and density, then its shape's data. The answer's values are each body's
    /// position x y z, rotation x y z w, linear and angular velocity, 13 values in the
    /// request's order, then the contacts touching after the last step, 9 values each: the two
    /// bodies' indices (the lower first), the deepest point x y z, the normal x y z from the
    /// first to the second, and their distance (negative while they overlap); its value is the
    /// contacts' count. The world is the request's alone: nothing carries over to the next.
    pub mod physics_op {
        pub const STEP: u32 = 0;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 1] = ["step"];
    }
    /// How a physics body moves: never; under gravity and contacts; or by its velocity alone,
    /// pushing what it meets.
    pub mod physics_body {
        pub const FIXED: u32 = 0;
        pub const DYNAMIC: u32 = 1;
        pub const KINEMATIC: u32 = 2;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 3] = ["fixed", "dynamic", "kinematic"];
    }
    /// A collider's shape, and what its size and data mean: a ball of radius x; a cuboid of
    /// half extents x y z; a capsule and a cylinder of radius x and half height y along their y
    /// axis; the convex hull of x points, its data x y z each; a heightfield of x columns by y
    /// rows of samples z metres apart, centred on its position, its data the heights row by row
    /// from -z, each row from -x; and a triangle mesh of x vertices and y triangles, its data
    /// the vertices x y z, then each triangle's three vertex indices. Coordinates, sizes,
    /// heights and speeds lie within a million metres (a second): a request past that fails.
    pub mod collider_shape {
        pub const BALL: u32 = 0;
        pub const CUBOID: u32 = 1;
        pub const CAPSULE: u32 = 2;
        pub const CYLINDER: u32 = 3;
        pub const CONVEX: u32 = 4;
        pub const HEIGHTFIELD: u32 = 5;
        pub const TRIMESH: u32 = 6;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 7] = ["ball", "cuboid", "capsule", "cylinder", "convex", "heightfield", "trimesh"];
    }
    /// Paths over a navmesh on the ground plane (x, z), triangulated from areas. The values are
    /// an agent's radius, then areas: each a kind (NAVMESH_AREA_*), a corner count and the
    /// corners' x z. The ground is walkable where a walkable area covers it and no obstacle
    /// does. find answers a short way from (`a`, `b`) to (`c`, `d`), its turns kept the radius
    /// off the corners they round and crossing no navmesh edge narrower than twice it, as x z
    /// points from start to goal, value their count; a start or goal off the navmesh moves to
    /// its nearest point, and none answers when the goal cannot be reached. triangles answers
    /// the walkable triangles, x z of each corner, value their count.
    pub mod navmesh_op {
        pub const FIND: u32 = 0;
        pub const TRIANGLES: u32 = 1;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 2] = ["find", "triangles"];
    }
    /// What a navmesh area is: ground to walk on, or something in the way.
    pub mod navmesh_area {
        pub const WALKABLE: u32 = 0;
        pub const OBSTACLE: u32 = 1;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 2] = ["walkable", "obstacle"];
    }
    /// Queries against `a` colliders, the request's first values (COLLIDER_SHAPE_*'s records
    /// with their data). raycast casts the rays that follow (origin x y z, direction x y z) up
    /// to `b` metres, a ray starting inside a shape hitting it at once while `c` is 1, and
    /// answers for each the collider it hits first (its index + 1, 0 for none), the distance
    /// and the surface's normal x y z, value the hits' count. overlap answers the indices of
    /// the colliders a query collider (the record that follows) touches, in order, value their
    /// count. nearest answers for each point that follows (x y z) the nearest collider within
    /// `b` metres (index + 1, 0 for none), the distance (0 inside) and its closest point x y z,
    /// value how many found one.
    pub mod spatial_op {
        pub const RAYCAST: u32 = 0;
        pub const OVERLAP: u32 = 1;
        pub const NEAREST: u32 = 2;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 3] = ["raycast", "overlap", "nearest"];
    }
    /// Deterministic simplex noise summed in octaves, the same on every host. The values start
    /// with its frequency (cycles a metre), amplitude, lacunarity (each octave's frequency over
    /// the last's), gain (each octave's amplitude over the last's) and style (NOISE_STYLE_*);
    /// `c` is the seed and `d` the octaves (1 to 16). heightfield answers a grid of `a` columns
    /// by `b` rows of heights, row by row, from the corner x z the next two values give, the
    /// last its spacing in metres (its rows along +z, its columns along +x). sample answers the
    /// height at each x z pair that follows. Heights stay within the amplitude; the value is
    /// their count.
    pub mod noise_op {
        pub const HEIGHTFIELD: u32 = 0;
        pub const SAMPLE: u32 = 1;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 2] = ["heightfield", "sample"];
    }
    /// How noise octaves add up: fractal Brownian motion (rolling hills), or ridged (sharp
    /// crests where the noise crosses zero).
    pub mod noise_style {
        pub const FBM: u32 = 0;
        pub const RIDGED: u32 = 1;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 2] = ["fbm", "ridged"];
    }
    /// A gamepad's buttons, in the order of the browser's standard gamepad: the four face
    /// buttons by where they sit (south is Xbox's A, PlayStation's cross), the shoulder
    /// buttons, the triggers (analog: their GamepadButton value runs from 0 to 1), select and
    /// start, the sticks pressed in, the directional pad and the home button.
    pub mod gamepad_button {
        pub const SOUTH: u32 = 0;
        pub const EAST: u32 = 1;
        pub const WEST: u32 = 2;
        pub const NORTH: u32 = 3;
        pub const LEFT_SHOULDER: u32 = 4;
        pub const RIGHT_SHOULDER: u32 = 5;
        pub const LEFT_TRIGGER: u32 = 6;
        pub const RIGHT_TRIGGER: u32 = 7;
        pub const SELECT: u32 = 8;
        pub const START: u32 = 9;
        pub const LEFT_STICK: u32 = 10;
        pub const RIGHT_STICK: u32 = 11;
        pub const DPAD_UP: u32 = 12;
        pub const DPAD_DOWN: u32 = 13;
        pub const DPAD_LEFT: u32 = 14;
        pub const DPAD_RIGHT: u32 = 15;
        pub const HOME: u32 = 16;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 17] = ["south", "east", "west", "north", "left_shoulder", "right_shoulder", "left_trigger", "right_trigger", "select", "start", "left_stick", "right_stick", "dpad_up", "dpad_down", "dpad_left", "dpad_right", "home"];
    }
    /// A gamepad's stick axes, in the order of the browser's standard gamepad: from -1 to 1,
    /// right and down positive.
    pub mod gamepad_axis {
        pub const LEFT_X: u32 = 0;
        pub const LEFT_Y: u32 = 1;
        pub const RIGHT_X: u32 = 2;
        pub const RIGHT_Y: u32 = 3;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 4] = ["left_x", "left_y", "right_x", "right_y"];
    }
    /// What moved an action (the Action event's device): the keyboard or mouse, a gamepad,
    /// Steam Input, or a touch screen. A game shows the prompts of the device the player used
    /// last.
    pub mod input_device {
        pub const KEYBOARD: u32 = 0;
        pub const GAMEPAD: u32 = 1;
        pub const STEAM: u32 = 2;
        pub const TOUCH: u32 = 3;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 4] = ["keyboard", "gamepad", "steam", "touch"];
    }
    /// What a binding reads: a key (its code, as KeyDown events carry it), a mouse button
    /// (numbered as PointerDown events number them), a gamepad button (GAMEPAD_BUTTON_*) or a
    /// gamepad axis (GAMEPAD_AXIS_*). A binding is three numbers: its source, its code and its
    /// scale, which multiplies what it reads (a key or button reads 1 held, an axis -1 to 1),
    /// so -1 makes a key push an axis action the other way and an axis bound with -1 reads its
    /// other half.
    pub mod input_source {
        pub const KEY: u32 = 0;
        pub const MOUSE: u32 = 1;
        pub const PAD_BUTTON: u32 = 2;
        pub const PAD_AXIS: u32 = 3;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 4] = ["key", "mouse", "pad_button", "pad_axis"];
    }
    /// The input service's operations, on action `a` (0 to MAX_ACTIONS - 1). declare declares
    /// it: `b` is 0 for a button (its value 1 while a binding holds it at half or more, else 0)
    /// or 1 for an axis (its value the bindings' sum, from -1 to 1), its values its name's `c`
    /// code points and then its default bindings, three numbers each (INPUT_SOURCE_*); it
    /// answers how many bindings it has, the player's if they rebound it. capture binds slot
    /// `b` to the next key, mouse button, gamepad button or stick the player presses (Escape
    /// gives up, answered failed), answering the binding as its values when it comes; cancel
    /// gives a capture up. bind sets slot `b` to the binding in the values, unbind removes slot
    /// `b`, reset gives the action its defaults back (every action's when `a` is MAX_ACTIONS),
    /// and bindings answers its bindings. The host keeps the player's bindings with their
    /// preferences.
    pub mod input_op {
        pub const DECLARE: u32 = 0;
        pub const CAPTURE: u32 = 1;
        pub const CANCEL: u32 = 2;
        pub const BIND: u32 = 3;
        pub const UNBIND: u32 = 4;
        pub const RESET: u32 = 5;
        pub const BINDINGS: u32 = 6;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 7] = ["declare", "capture", "cancel", "bind", "unbind", "reset", "bindings"];
    }
    /// Where a touch is (the Touch event's phase): put down, moved, lifted, or taken away by
    /// the system (a gesture it claimed, the window losing the focus).
    pub mod touch_phase {
        pub const START: u32 = 0;
        pub const MOVE: u32 = 1;
        pub const END: u32 = 2;
        pub const CANCEL: u32 = 3;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 4] = ["start", "move", "end", "cancel"];
    }
    /// Where each number sits in a devtools packets answer's frame: its position; its input's
    /// events and lanes; its render packet's lanes; its settings (camera, light, sky, quality
    /// and the rest, one element whose fields a diff counts); its draws, solids, decals,
    /// sounds, sprites, insets, field patches, interface nodes, service requests, buses and
    /// voices; and its requests to save, load, save and load the scene, checkpoint, undo, redo
    /// and exit, a bit each from 1 up.
    pub mod devtools_packet {
        pub const POSITION: u32 = 0;
        pub const EVENTS: u32 = 1;
        pub const INPUT: u32 = 2;
        pub const PACKET: u32 = 3;
        pub const SETTINGS: u32 = 4;
        pub const DRAWS: u32 = 5;
        pub const SOLIDS: u32 = 6;
        pub const DECALS: u32 = 7;
        pub const SOUNDS: u32 = 8;
        pub const SPRITES: u32 = 9;
        pub const INSETS: u32 = 10;
        pub const FIELDS: u32 = 11;
        pub const UI: u32 = 12;
        pub const SERVICES: u32 = 13;
        pub const BUSES: u32 = 14;
        pub const VOICES: u32 = 15;
        pub const REQUESTS: u32 = 16;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 17] = ["position", "events", "input", "packet", "settings", "draws", "solids", "decals", "sounds", "sprites", "insets", "fields", "ui", "services", "buses", "voices", "requests"];
    }
    /// How a devtools watch compares a lane of the game's state with its value: equal to it,
    /// not equal, below it, above it, or (the value ignored) changed from what the lane held
    /// when the watch started.
    pub mod devtools_compare {
        pub const EQUAL: u32 = 0;
        pub const UNEQUAL: u32 = 1;
        pub const BELOW: u32 = 2;
        pub const ABOVE: u32 = 3;
        pub const CHANGES: u32 = 4;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 5] = ["equal", "unequal", "below", "above", "changes"];
    }
    /// The kinds of the game's catalog a devtools assets, preview or swap request names.
    pub mod devtools_asset {
        pub const MESH: u32 = 0;
        pub const MATERIAL: u32 = 1;
        pub const SOUND: u32 = 2;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 3] = ["mesh", "material", "sound"];
    }
    /// Where a catalog asset stands in the host: not asked for yet, loading, ready to use, or
    /// failed to load (its last good revision, if any, still used).
    pub mod devtools_asset_state {
        pub const UNUSED: u32 = 0;
        pub const LOADING: u32 = 1;
        pub const READY: u32 = 2;
        pub const FAILED: u32 = 3;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 4] = ["unused", "loading", "ready", "failed"];
    }
    /// The engine's own preferences, the same for every game: the player sets them through the
    /// host (its options, the platform's settings) or a game's settings (the preferences
    /// service), and the host keeps them. color_filter is a COLOR_FILTER_* the renderer applies
    /// to the whole picture, filter_strength how much of it (0 to 1), text_scale how much
    /// larger the game draws its text (1 to 2) and reduced_motion 1 when the player wants as
    /// little motion as can be (camera flights, shakes, slides and flashes left out). They
    /// reach the game as Preference events at index ENGINE_PREFERENCES plus the preference,
    /// played back as the game's own are.
    pub mod engine_preference {
        pub const COLOR_FILTER: u32 = 0;
        pub const FILTER_STRENGTH: u32 = 1;
        pub const TEXT_SCALE: u32 = 2;
        pub const REDUCED_MOTION: u32 = 3;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 4] = ["color_filter", "filter_strength", "text_scale", "reduced_motion"];
    }
    /// A colour filter for colour vision deficiency: none, or the colours shifted so that those
    /// a player without red (protanopia), green (deuteranopia) or blue (tritanopia) cones
    /// confuses tell apart again; achromatopsia shows the picture in greys of its luminance.
    pub mod color_filter {
        pub const NONE: u32 = 0;
        pub const PROTANOPIA: u32 = 1;
        pub const DEUTERANOPIA: u32 = 2;
        pub const TRITANOPIA: u32 = 3;
        pub const ACHROMATOPSIA: u32 = 4;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 5] = ["none", "protanopia", "deuteranopia", "tritanopia", "achromatopsia"];
    }
    /// The engine's preferences (ENGINE_PREFERENCE_*): set stores preference `a` as `b`, which
    /// the host applies at once and plays back to the game from then on, and answers the value
    /// it kept (clamped to the preference's range), or failed for a preference or value it does
    /// not take.
    pub mod preferences_op {
        pub const SET: u32 = 0;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 1] = ["set"];
    }
    /// The player's language, from the game's string tables (its text.json, and
    /// locales/TAG.json for each language). text answers the string whose key is the request's
    /// text in the player's language, laid out to draw left to right (right-to-left scripts
    /// reordered and Arabic joined), with `a` as its count when `b` is 1 (choosing the plural
    /// form and filling {count}); value 1 when it reads right to left, unavailable when no
    /// table has the key; where the text drawn differs from the text read (right to left), its
    /// values are the text drawn, a 0, then the text in reading order, unshaped, for assistive
    /// technology. language answers the language in use as a BCP 47 tag, value 1 for a
    /// right-to-left one. plural answers the plural category (PLURAL_CATEGORY_*) of count `a`
    /// in it as its value.
    pub mod locale_op {
        pub const TEXT: u32 = 0;
        pub const LANGUAGE: u32 = 1;
        pub const PLURAL: u32 = 2;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 3] = ["text", "language", "plural"];
    }
    /// The plural categories of Unicode's CLDR, which choose a string's form for a count.
    pub mod plural_category {
        pub const ZERO: u32 = 0;
        pub const ONE: u32 = 1;
        pub const TWO: u32 = 2;
        pub const FEW: u32 = 3;
        pub const MANY: u32 = 4;
        pub const OTHER: u32 = 5;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 6] = ["zero", "one", "two", "few", "many", "other"];
    }
    /// What an accessibility audit found: a node assistive technology cannot name (a control or
    /// an image without a name), a focus problem (a trap: a modal dialog with nothing to focus,
    /// or the keyboard unable to leave; a focus on a node the tree leaves out; two focused
    /// nodes), text whose contrast with what lies behind it is below WCAG's 4.5:1 (3:1 for
    /// large text), or a control smaller than 24 by 24 pixels.
    pub mod audit_finding {
        pub const UNNAMED: u32 = 0;
        pub const FOCUS: u32 = 1;
        pub const CONTRAST: u32 = 2;
        pub const SIZE: u32 = 3;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 4] = ["unnamed", "focus", "contrast", "size"];
    }
    /// The captions the game's sounds carry (the assets' captions.json, by sound): text answers
    /// the caption of clip `a` as text, its value the seconds it shows for, or unavailable when
    /// the clip has none; active answers the captions showing now, newest first, one a line,
    /// its value how many.
    pub mod captions_op {
        pub const TEXT: u32 = 0;
        pub const ACTIVE: u32 = 1;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 2] = ["text", "active"];
    }
    /// Where each number sits at the start of a devtools audio answer: the output's latency in
    /// milliseconds, from a frame's sound reaching the mixer to it leaving the speakers (-1
    /// when the host cannot tell); the sample rate; the device's buffer, in samples a callback,
    /// as asked for or as the system chose (-1 while unknown); the voices and sounds playing,
    /// and those of them streamed; how often a stream ran dry so far; and how many buses
    /// follow.
    pub mod devtools_audio {
        pub const LATENCY: u32 = 0;
        pub const RATE: u32 = 1;
        pub const BUFFER: u32 = 2;
        pub const VOICES: u32 = 3;
        pub const STREAMS: u32 = 4;
        pub const UNDERRUNS: u32 = 5;
        pub const BUSES: u32 = 6;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 7] = ["latency", "rate", "buffer", "voices", "streams", "underruns", "buses"];
    }
    /// Each bus's numbers in a devtools audio answer: its output's peak and RMS on the left and
    /// right since the last answer, and its state: 1 when a frame uses it, plus 2 when muted,
    /// plus 4 when soloed.
    pub mod devtools_bus {
        pub const PEAK_LEFT: u32 = 0;
        pub const PEAK_RIGHT: u32 = 1;
        pub const RMS_LEFT: u32 = 2;
        pub const RMS_RIGHT: u32 = 3;
        pub const STATE: u32 = 4;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 5] = ["peak_left", "peak_right", "rms_left", "rms_right", "state"];
    }
    /// How a panner's gain falls with distance, as WebAudio's PannerNode distanceModel: linear
    /// from 1 at ref_distance to 1 - rolloff at max_distance; inverse, ref_distance /
    /// (ref_distance + rolloff * (distance - ref_distance)); exponential, (distance /
    /// ref_distance) ^ -rolloff; nearer than ref_distance is 1.
    pub mod distance_model {
        pub const LINEAR: u32 = 0;
        pub const INVERSE: u32 = 1;
        pub const EXPONENTIAL: u32 = 2;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 3] = ["linear", "inverse", "exponential"];
    }
    /// Crash and performance reports for the game's developer, which the player opts into:
    /// consent answers what the player agreed to as its value, 1 for crash reports plus 2 for
    /// performance reports (0 for neither, as before they are asked); set_consent sets it to
    /// `a`, the same sum; queued answers how many reports wait to be sent. Each answers
    /// unavailable where nothing could be sent (a package whose developer gave no address for
    /// reports, a checkout, the browser), so that the game offers no choice with nothing behind
    /// it.
    pub mod reports_op {
        pub const CONSENT: u32 = 0;
        pub const SET_CONSENT: u32 = 1;
        pub const QUEUED: u32 = 2;
        /// The names of the codes, by code.
        pub const NAMES: [&str; 3] = ["consent", "set_consent", "queued"];
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
        /// A gamepad was connected: its number, and 1 if its buttons and axes are laid out as
        /// the standard gamepad (GAMEPAD_BUTTON_*, GAMEPAD_AXIS_*), 0 if they come in the
        /// device's own order.
        GamepadConnected { pad: u32, standard: u32 },
        /// A gamepad went away: its held buttons and moved axes came back first.
        GamepadDisconnected { pad: u32 },
        /// A gamepad button (GAMEPAD_BUTTON_*) changed: the pad, its value from 0 to 1, and 1
        /// while it is pressed (at half or more).
        GamepadButton { button: u32, pad: u32, value: f32, pressed: u32 },
        /// A gamepad axis (GAMEPAD_AXIS_*) moved: the pad and its value from -1 to 1, past the
        /// stick's dead zone (inside it, 0).
        GamepadAxis { axis: u32, pad: u32, value: f32 },
        /// An action the game declared (the input service) changed: which, its value (a
        /// button's 1 or 0, an axis from -1 to 1) and the device that moved it
        /// (INPUT_DEVICE_*).
        Action { action: u32, value: f32, device: u32 },
        /// A character of text, typed, composed or committed by an input method, one code point
        /// an event, in order; or with composing 1, one of the composition's characters after
        /// its Composition event.
        TextInput { character: u32, composing: u32 },
        /// An input method's composition (the text being composed, before it is committed)
        /// changed: its length in code points, which follow as TextInput events with composing
        /// 1, and the caret's place in it. A length of 0 ends it.
        Composition { length: u32, cursor: u32 },
        /// How far the mouse moved while the pointer is locked, unaccelerated where the
        /// platform allows: the device's counts natively, CSS pixels in a browser.
        MouseMotion { dx: f32, dy: f32 },
        /// A touch on the screen, by number (from 0, reused once lifted), where it is in window
        /// pixels, and its phase (TOUCH_PHASE_*). The first touch down also moves and presses
        /// the pointer (button 0), for games that only take the pointer.
        Touch { id: u32, x: f32, y: f32, phase: u32 },
        /// Touches dragged together: how far their centre moved, in window pixels, and how many
        /// there are.
        Pan { touches: u32, dx: f32, dy: f32 },
        /// Two touches moved apart or together: their centre, in window pixels, and the factor
        /// their distance grew by since the last Pinch.
        Pinch { x: f32, y: f32, scale: f32 },
        /// In a net session (pen's docs/netplay.md) the events after it, up to the next Player,
        /// are this player's, numbered from 0 up to the players the game declares; every frame
        /// lists each player in order, events or none. Outside one there are none, and every
        /// event is player 0's.
        Player { index: u32 },
        /// The host session started, once after creating the game or restoring its initial save
        /// and before live frames. restored is 1 when the initial save matched the game's state
        /// schema and was restored, otherwise 0. Reloads, loads, rewinds and replays do not
        /// send it; it is not recorded as input.
        SessionStarted { restored: u32 },
        /// Once during an opted-in loading presentation after its scene, assets, pipelines and
        /// texture fades are ready. Delivered as a normal event so game-owned loading
        /// presentations can finish deterministically.
        RendererReady,
        /// One of the game's own events, which its pen.json `protocol` declares: its tag there,
        /// and up to three values. Hosts send them only to a game that declares them, whose
        /// generated pen/extensions.sq decodes them; pen's own bindings leave them out.
        Game { event: u32, a: f32, b: f32, c: f32 },
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
        /// Composite after this many frame sprites; zero draws below all sprites.
        pub after_sprites: u32,
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
        /// Linear compressor threshold in (0, 1]; 0 disables bus compression.
        pub comp_threshold: f32,
        /// Compressor ratio (>= 1.0); values <= 1.0 disable bus compression.
        pub comp_ratio: f32,
        /// Bus index whose RMS level ducks this bus when `duck_amount > 0`.
        pub duck_bus: u32,
        /// Sidechain ducking attenuation share in 0..1; 0 disables ducking.
        pub duck_amount: f32,
        /// Low-shelf EQ gain at 200 Hz in dB; 0 bypasses.
        pub eq_low_db: f32,
        /// Peaking EQ gain at 1000 Hz (Q=1.0) in dB; 0 bypasses.
        pub eq_mid_db: f32,
        /// High-shelf EQ gain at 4000 Hz in dB; 0 bypasses.
        pub eq_high_db: f32,
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
    /// A layer of a skinned draw's pose: clip `clip` of the draw's mesh at `time` seconds,
    /// counting `weight` among the draw's layers.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Pose {
        /// The draw it poses: its place in the frame's draws.
        pub draw: u32,
        /// Which of the mesh's clips, in the order its file lists them; one the mesh lacks
        /// counts for nothing.
        pub clip: u32,
        /// Seconds into the clip, round it as often as need be.
        pub time: f32,
        /// How much it counts among its draw's layers: 0 or more.
        pub weight: f32,
        /// Which draw list `draw` indexes: 0 for the main scene's `draws`, or `1 + k` for
        /// `insets[k].draws`.
        pub inset: u32,
    }
    /// A particle emitter: `rate` particles a second, each living `lifetime` seconds, born in a
    /// ball round `transform`'s origin and flying from it under gravity and drag, sized and
    /// colored over their lives.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Emitter {
        /// Its texture (round soft dots without one) and color, which tint every particle.
        pub material: u32,
        /// Where particles are born, and how their velocity turns.
        pub transform: Matrix,
        /// What its particles' randomness follows from.
        pub seed: u32,
        /// Seconds since it began: particles are born `rate` a second from 0 on, each the same
        /// whatever the frame rate.
        pub time: f32,
        /// Seconds it gives birth for; 0 for as long as frames keep it.
        pub duration: f32,
        /// Particles born a second.
        pub rate: f32,
        /// Seconds each particle lives.
        pub lifetime: f32,
        /// Particles are born anywhere in the ball of this radius round the emitter.
        pub radius: f32,
        /// Each particle's velocity at birth, in the emitter's space.
        pub velocity: Vector,
        /// A random velocity up to this fast, in any direction, added to each at birth.
        pub spread: f32,
        /// Acceleration in world space, units a second a second.
        pub gravity: Vector,
        /// How fast velocity fades, a share a second: 0 for none.
        pub drag: f32,
        /// Particles' width in world units at birth, changing evenly to `size_end` at death.
        pub size_start: f32,
        pub size_end: f32,
        /// Particles' color at birth, changing evenly to `color_end` at death; alpha fades
        /// them.
        pub color_start: Color,
        pub color_end: Color,
        /// Particles add their light to what is behind them, rather than covering it.
        pub additive: bool,
        /// View-space fade distance in world units against opaque scene depth; 0 disables soft
        /// particle fading.
        pub soft_depth: f32,
        /// Directional sun and ambient lighting weight on particles (0 unlit, 1 fully lit).
        pub lit: f32,
        /// Horizontal sprite-sheet frames in the emitter texture (0 or 1 for a single frame).
        pub flipbook_cols: u32,
        /// Vertical sprite-sheet frames in the emitter texture (0 or 1 for a single frame).
        pub flipbook_rows: u32,
        /// Flipbook playback rate in frames per second (0 plays one full sheet over each
        /// particle's lifetime).
        pub flipbook_fps: f32,
    }
    /// A point light, or a spot light shining along `direction` within a cone, lighting what
    /// lies within `radius` of `position`.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Light {
        pub position: Vector,
        /// How far it reaches: its light falls smoothly to nothing there.
        pub radius: f32,
        /// Its color times its intensity.
        pub color: Vector,
        /// The way a spot light shines, a unit vector; a point light's is ignored.
        pub direction: Vector,
        /// The cosine of the half-angle within which a spot light is at full strength.
        pub inner: f32,
        /// The cosine of the half-angle of a spot light's cone, where its light ends; -1 for a
        /// point light, which shines all round.
        pub outer: f32,
        /// Whether it asks to cast shadows; the host picks which of those do, within
        /// `shadow.local`.
        pub shadow: bool,
        /// A number that stays with the light from frame to frame: it breaks ties between
        /// equally large casters and names the shadow map the host may keep.
        pub id: u32,
    }
    /// One sun shadow map.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Cascade {
        /// The map's orthographic window, depth range included, in the light's view.
        pub projection: Matrix,
        /// A map texel's width in world units.
        pub texel: f32,
        /// How far ahead of the camera, along its view, the slice this map covers ends.
        pub far: f32,
    }
    /// A diffuse light probe: nine spherical-harmonic RGB terms of the light there, and whether
    /// it is valid.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Probe {
        /// Term 0 (constant) of the light a white surface facing each way shows here: cosine
        /// lobe applied, divided by pi.
        pub sh0: Vector,
        /// Term 1 (y) of the light a white surface facing each way shows here: cosine lobe
        /// applied, divided by pi.
        pub sh1: Vector,
        /// Term 2 (z) of the light a white surface facing each way shows here: cosine lobe
        /// applied, divided by pi.
        pub sh2: Vector,
        /// Term 3 (x) of the light a white surface facing each way shows here: cosine lobe
        /// applied, divided by pi.
        pub sh3: Vector,
        /// Term 4 (xy) of the light a white surface facing each way shows here: cosine lobe
        /// applied, divided by pi.
        pub sh4: Vector,
        /// Term 5 (yz) of the light a white surface facing each way shows here: cosine lobe
        /// applied, divided by pi.
        pub sh5: Vector,
        /// Term 6 (zz) of the light a white surface facing each way shows here: cosine lobe
        /// applied, divided by pi.
        pub sh6: Vector,
        /// Term 7 (xz) of the light a white surface facing each way shows here: cosine lobe
        /// applied, divided by pi.
        pub sh7: Vector,
        /// Term 8 (xx_yy) of the light a white surface facing each way shows here: cosine lobe
        /// applied, divided by pi.
        pub sh8: Vector,
        /// 1 for a probe in open space, 0 for one inside geometry, whose light the host leaves
        /// out.
        pub valid: f32,
    }
    /// A local reflection probe: where it is captured from, the box it lights, and the box its
    /// surroundings are projected onto.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct ReflectionProbe {
        /// A number that stays with the probe from frame to frame: its cube is kept under it,
        /// and it breaks ties.
        pub id: u32,
        /// Which state of the surroundings the game wants captured; the host draws the cube
        /// again when this differs from the capture it holds.
        pub revision: u32,
        /// Where boxes overlap, a higher priority takes the surface's reflection before lower
        /// ones share it; equal priorities blend.
        pub priority: u32,
        /// Asks for a new capture: the host draws the cube again when this rises from 0 to 1.
        pub capture: bool,
        /// Where the cube is drawn from, in world units.
        pub position: Vector,
        /// The middle of the influence box.
        pub center: Vector,
        /// Half the influence box's size along x, y and z: surfaces inside it reflect this
        /// probe.
        pub extent: Vector,
        /// How far inside the box's faces the probe fades in, from 0 at a face to 1 at that
        /// distance; 0 for a hard edge.
        pub blend: f32,
        /// The middle of the box the surroundings are taken to be, for box projection.
        pub parallax_center: Vector,
        /// Half the size of the box the cube is projected onto, which should hold the room the
        /// probe sits in; any component 0 or less for no projection (the cube is taken as
        /// infinitely far).
        pub parallax_extent: Vector,
        /// How far the capture sees: the farthest of its draws, in world units.
        pub far: f32,
    }
    /// A local fog volume.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct FogVolume {
        /// 0 a box, 1 an ellipsoid.
        pub shape: u32,
        /// The middle of the volume, in world units.
        pub center: Vector,
        /// Half the volume's size along x, y and z (an ellipsoid's semi-axes).
        pub extent: Vector,
        /// The extinction at full strength, per world unit.
        pub density: f32,
        /// The share of light it scatters, by channel; the rest it absorbs.
        pub albedo: Vector,
        /// The light it gives out, as radiance per unit of density (so a thicker volume glows
        /// more); 0 for none.
        pub emission: Vector,
        /// The share of the half size, from the edge in, over which the density rises smoothly
        /// from nothing to full (0 to 1; 0 for a hard edge).
        pub falloff: f32,
    }
    /// A listener: where it is, the way it faces and its up, in world units.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Listener {
        pub position: Vector,
        /// The way it faces.
        pub forward: Vector,
        /// Its up, which with `forward` tells left from right.
        pub up: Vector,
        /// World-space listener velocity in world units per second, for Doppler pitch shifting.
        pub velocity: Vector,
    }
    /// Where a voice or sound is, as a PannerNode places it: from the listener, equal-power
    /// panned by its azimuth (in place of its pan), and quieted by its distance model and its
    /// cone. Positions and orientations glide as gains do (AUDIO_GLIDE_MS); the rest change at
    /// once. A voice that gains or loses its panner starts afresh, as one whose bus changed; a
    /// panner that places nothing is ignored, and of two for one voice or sound hosts take the
    /// first.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Panner {
        /// The voice it places, by id, or 0 for a sound.
        pub voice: u32,
        /// With `voice` 0, the sound it places, by its place in the frame's `sounds`.
        pub sound: u32,
        pub position: Vector,
        /// The way its cone faces; zero for none, heard alike from every side.
        pub orientation: Vector,
        /// How its gain falls with distance (DISTANCE_MODEL_*).
        pub distance_model: u32,
        /// Within it, distance does not quiet it.
        pub ref_distance: f32,
        /// Beyond it, the linear model quiets it no further.
        pub max_distance: f32,
        /// How fast distance quiets it.
        pub rolloff: f32,
        /// The angle, in degrees, of the cone it is heard whole within.
        pub cone_inner: f32,
        /// The angle, in degrees, beyond which it is heard at cone_outer_gain.
        pub cone_outer: f32,
        pub cone_outer_gain: f32,
        /// Enable HRTF binaural spatialization (ITD delay and head-shadowing ILD natively,
        /// PanningModelType::Hrtf in WebAudio) instead of equal-power panning.
        pub hrtf: bool,
        /// World-space source velocity in world units per second, for Doppler pitch shifting.
        pub velocity: Vector,
        /// Doppler pitch shift multiplier (0 disables, 1 uses physical speed of sound 343 m/s).
        pub doppler: f32,
        /// Direct-path acoustic occlusion in 0..1 (0 open, 1 fully muffled by lowpass and
        /// direct gain attenuation).
        pub occlusion: f32,
        /// Optional acoustic reverb zone ID (0 auto-detects from world position against
        /// reverb_zones).
        pub zone: u32,
    }
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Instance {
        pub id_low: u32,
        pub id_high: u32,
        pub order: u32,
        pub mesh: u32,
        pub material: u32,
        pub transform: Matrix,
        /// Optional four shader values, overriding the material's instance defaults.
        pub parameters: Option<[f32; 4]>,
    }
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct InstanceRemoval {
        pub id_low: u32,
        pub id_high: u32,
    }
    /// One explicit joint skinning matrix for a skinned draw.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct BoneMatrix {
        /// Which draw list `draw` indexes: 0 for the main scene's `draws`, or `1 + k` for
        /// `insets[k].draws`.
        pub inset: u32,
        /// The draw this joint matrix skins.
        pub draw: u32,
        /// Joint index within the mesh's skin palette.
        pub joint: u32,
        /// Column-major skinning matrix (world joint transform times inverse bind matrix).
        pub matrix: Matrix,
    }
    /// One morph target blend weight for a draw.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct MorphWeight {
        /// Which draw list `draw` indexes: 0 for the main scene's `draws`, or `1 + k` for
        /// `insets[k].draws`.
        pub inset: u32,
        /// The draw whose mesh vertices are morphed.
        pub draw: u32,
        /// Morph target index in the mesh's morph target array.
        pub target: u32,
        /// Linear blend weight for this target.
        pub weight: f32,
    }
    /// Previous-frame model transform for a moving draw.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Motion {
        /// The scene draw index in `frame.draws`.
        pub draw: u32,
        /// The draw's column-major model transform in the previous frame.
        pub previous: Matrix,
    }
    /// One spine control point of a camera-facing ribbon trail.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct RibbonPoint {
        /// Ribbon trail ID; consecutive points sharing an ID, material and blend mode form one
        /// strip.
        pub ribbon: u32,
        /// Material providing color tint and optional texture.
        pub material: u32,
        /// Ribbon adds light to what is behind it rather than alpha-blending.
        pub additive: bool,
        /// World-space spine point.
        pub position: Vector,
        /// Ribbon width in world units at this control point.
        pub width: f32,
        /// RGBA color tint and opacity at this control point.
        pub color: Color,
        /// View-space soft depth fade distance in world units; 0 disables.
        pub soft_depth: f32,
    }
    /// Dual-motor haptic feedback command for one gamepad.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Haptic {
        /// Gamepad index (0 to MAX_GAMEPADS - 1).
        pub pad: u32,
        /// Low-frequency (strong) motor intensity in 0..1.
        pub low: f32,
        /// High-frequency (weak) motor intensity in 0..1.
        pub high: f32,
        /// Pulse duration in seconds (0 stops rumble or applies for one frame).
        pub duration: f32,
    }
    /// One spatial acoustic reverb zone and portal boundary.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct ReverbZone {
        /// Zone identifier (1..64; 0 matches unzoned space).
        pub id: u32,
        /// World-space AABB center of the acoustic zone.
        pub position: Vector,
        /// World-space half extents of the zone box.
        pub half_extents: Vector,
        /// Outer transition distance in world units for smooth room blending.
        pub blend: f32,
        /// RT60 reverb decay time in seconds for this zone.
        pub decay: f32,
        /// High-frequency reverb damping in 0..1 for this zone.
        pub damping: f32,
        /// Reverb send wet multiplier for sources inside this zone (0 defaults to 1).
        pub wet: f32,
        /// Extra acoustic occlusion in 0..1 applied across room boundaries/portals when source
        /// and listener occupy different zones.
        pub portal_attenuation: f32,
    }
    /// One real-time procedural or granular DSP synthesizer voice.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct SynthVoice {
        /// Persistent synth voice identifier.
        pub id: u32,
        /// Oscillator/synth mode: 0 Sine, 1 Saw, 2 Square, 3 Triangle, 4 Noise, 5 Granular
        /// cloud.
        pub mode: u32,
        /// Target mixer bus index (0..MAX_BUSES-1).
        pub bus: u32,
        /// Base oscillator frequency in Hz (or grain carrier frequency in granular mode).
        pub frequency: f32,
        /// Linear output gain.
        pub gain: f32,
        /// Stereo pan in -1..1.
        pub pan: f32,
        /// Pitch detune in cents (or granular pitch jitter spread in cents).
        pub detune: f32,
        /// ADSR envelope attack time in seconds.
        pub attack: f32,
        /// ADSR envelope decay time in seconds.
        pub decay: f32,
        /// ADSR envelope sustain level in 0..1.
        pub sustain: f32,
        /// ADSR envelope release time in seconds when the voice leaves the frame.
        pub release: f32,
        /// Resonant lowpass filter cutoff frequency in Hz.
        pub cutoff: f32,
        /// Resonant lowpass filter Q factor (0.5..20).
        pub resonance: f32,
        /// Granular window duration in seconds (used in Granular mode).
        pub grain_size: f32,
        /// Granular harmonic/playback pitch multiplier (used in Granular mode).
        pub grain_pitch: f32,
        /// Reverb send level.
        pub reverb: f32,
    }
    /// One LTC area light or cookie projector.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct AreaLightSpec {
        /// Area light shape: 0 Rect, 1 Disk, 2 Tube, 3 Cookie projector.
        pub kind: u32,
        /// World-space center position.
        pub position: Vector,
        /// Unit local right axis.
        pub right: Vector,
        /// Unit local up axis (light emits along cross(right, up)).
        pub up: Vector,
        /// Linear RGB light color.
        pub color: Vector,
        /// Radiant intensity / luminance multiplier.
        pub intensity: f32,
        /// Full width (or diameter for disk, length for tube) in world units.
        pub width: f32,
        /// Full height (or tube radius) in world units.
        pub height: f32,
        /// Maximum attenuation distance in world units.
        pub range: f32,
        /// 1-based catalog material index whose texture acts as a 2D light cookie (0 for none).
        pub cookie_material: u32,
        /// Emit from both front and back faces.
        pub two_sided: bool,
    }
    /// One PBR deferred decal volume.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct PbrDecalSpec {
        /// Catalog material supplying decal albedo, normal, ORM, and emissive maps.
        pub material: u32,
        /// World-space oriented unit-cube decal projection transform.
        pub transform: Matrix,
        /// Albedo RGBA tint and opacity.
        pub albedo: Color,
        /// Tangent-space normal blending weight in 0..1.
        pub normal_strength: f32,
        /// Decal roughness factor in 0..1.
        pub roughness: f32,
        /// Decal metallic factor in 0..1.
        pub metallic: f32,
        /// Additive emissive intensity multiplier.
        pub emissive_strength: f32,
        /// Normal angle fade cutoff cosine in 0..1.
        pub fade: f32,
    }
    /// One render-to-texture or split-screen secondary camera.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct CameraTargetSpec {
        /// Catalog material whose texture key receives the rendered output (ignored when
        /// split_screen is true).
        pub material: u32,
        /// Offscreen target width in pixels (0 defaults to 256).
        pub width: u32,
        /// Offscreen target height in pixels (0 defaults to 256).
        pub height: u32,
        /// Normalized [0,1] split-screen X origin.
        pub viewport_x: f32,
        /// Normalized [0,1] split-screen Y origin.
        pub viewport_y: f32,
        /// Normalized [0,1] split-screen width.
        pub viewport_w: f32,
        /// Normalized [0,1] split-screen height.
        pub viewport_h: f32,
        /// When true, composites this camera directly onto the main screen at the normalized
        /// viewport rect.
        pub split_screen: bool,
        /// Secondary camera column-major view matrix.
        pub view: Matrix,
        /// Secondary camera column-major projection matrix.
        pub projection: Matrix,
    }
    /// One GPU video texture playback directive.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct VideoStreamSpec {
        /// Catalog material whose texture key names the video asset (.y4m/.ivf).
        pub material: u32,
        /// Target playback timestamp in seconds.
        pub time: f32,
        /// Whether playback is active.
        pub playing: bool,
        /// Whether playback loops at end of stream.
        pub looping: bool,
        /// Playback speed multiplier (0 defaults to 1.0).
        pub speed: f32,
    }
    /// How the game wants this frame drawn: samples per pixel, FXAA, render scale, texture
    /// filtering, ambient occlusion, a sky, screen effects, bloom and temporal antialiasing.
    #[derive(Clone, Copy, Debug, PartialEq)]
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
        /// Temporal antialiasing: each frame's samples move within their pixels and blend with
        /// the frames before, followed through the camera's motion.
        pub taa: bool,
        /// Screen-space reflections of glossy surfaces: 0 off, 1 low (a short march at a coarse
        /// step), 2 high (a longer march, finer refinement). They read the last frame's
        /// resolved picture and the camera's depth, so they need a perspective camera, and they
        /// turn temporal antialiasing on: its history resolves their noise.
        pub ssr: u32,
        /// The froxel volume's tier: 0 off (the mist is analytic, with no shafts or local
        /// volumes), 1 low (a coarser grid, fewer slices and sun-shadow taps), 2 high.
        pub volumetrics: u32,
        /// Screen-space diffuse global illumination bounce over the opaque scene.
        pub ssgi: bool,
        /// Weighted Blended Order-Independent Transparency for overlapping translucent draws.
        pub oit: bool,
        /// Hillaire 2020 physical atmosphere LUTs and raymarched 3D volumetric clouds.
        pub physical_sky: bool,
        /// TAAU temporal upscaling and RCAS sharpening when render scale < 1.
        pub upscale: bool,
        /// GPU BVH ray tracing tier: 0 off, 1 RT sun shadows, 2 RT shadows + RTAO, 3 RT shadows
        /// + RTAO + RT reflections/GI.
        pub ray_tracing: u32,
        /// Dynamic Diffuse Global Illumination octahedral probe volume evaluation.
        pub ddgi: bool,
    }
    impl Default for Quality {
        fn default() -> Self {
            Self {
                msaa: Default::default(),
                fxaa: Default::default(),
                scale: 1.0,
                filtering: 1,
                occlusion: Default::default(),
                sky: Default::default(),
                effects: Default::default(),
                bloom: Default::default(),
                taa: Default::default(),
                ssr: Default::default(),
                volumetrics: Default::default(),
                ssgi: Default::default(),
                oit: Default::default(),
                physical_sky: Default::default(),
                upscale: Default::default(),
                ray_tracing: 0,
                ddgi: Default::default(),
            }
        }
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
        /// Per-channel scale turning the sky's zenith, horizon and ground colors into the light
        /// it gives all round (the radiance behind the sh coefficients and glossy reflections);
        /// 0 for no reflections.
        pub tint: Vector,
        /// Coefficient 0 (real spherical harmonic constant, bands 0 to 2) of the sky's diffuse
        /// light: what a white surface facing each way shows, in linear color, the cosine lobe
        /// already applied and divided by pi.
        pub sh0: Vector,
        /// Coefficient 1 (real spherical harmonic y, bands 0 to 2) of the sky's diffuse light:
        /// what a white surface facing each way shows, in linear color, the cosine lobe already
        /// applied and divided by pi.
        pub sh1: Vector,
        /// Coefficient 2 (real spherical harmonic z, bands 0 to 2) of the sky's diffuse light:
        /// what a white surface facing each way shows, in linear color, the cosine lobe already
        /// applied and divided by pi.
        pub sh2: Vector,
        /// Coefficient 3 (real spherical harmonic x, bands 0 to 2) of the sky's diffuse light:
        /// what a white surface facing each way shows, in linear color, the cosine lobe already
        /// applied and divided by pi.
        pub sh3: Vector,
        /// Coefficient 4 (real spherical harmonic xy, bands 0 to 2) of the sky's diffuse light:
        /// what a white surface facing each way shows, in linear color, the cosine lobe already
        /// applied and divided by pi.
        pub sh4: Vector,
        /// Coefficient 5 (real spherical harmonic yz, bands 0 to 2) of the sky's diffuse light:
        /// what a white surface facing each way shows, in linear color, the cosine lobe already
        /// applied and divided by pi.
        pub sh5: Vector,
        /// Coefficient 6 (real spherical harmonic zz, bands 0 to 2) of the sky's diffuse light:
        /// what a white surface facing each way shows, in linear color, the cosine lobe already
        /// applied and divided by pi.
        pub sh6: Vector,
        /// Coefficient 7 (real spherical harmonic xz, bands 0 to 2) of the sky's diffuse light:
        /// what a white surface facing each way shows, in linear color, the cosine lobe already
        /// applied and divided by pi.
        pub sh7: Vector,
        /// Coefficient 8 (real spherical harmonic xx_yy, bands 0 to 2) of the sky's diffuse
        /// light: what a white surface facing each way shows, in linear color, the cosine lobe
        /// already applied and divided by pi.
        pub sh8: Vector,
    }
    /// The participating medium: a mist beyond `radius` of the origin whose density falls off
    /// exponentially with height, drawn through a froxel volume out to `distance` (lit by the
    /// sun, the lights and the sky, with shafts and shadows) and analytically beyond it;
    /// density 0 and no fog volumes for none.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Fog {
        pub radius: f32,
        pub density: f32,
        /// The scale height of the mist: its density falls by a factor of e over this many
        /// world units above the ground (above y = 0).
        pub height: f32,
        /// How far past `radius` (world units) the mist takes to rise from nothing to its full
        /// density, smoothly; 0 for a hard edge.
        pub edge: f32,
        /// How far ahead of the camera the froxel volume reaches, along the view; beyond it the
        /// mist continues analytically with the same medium. 0 draws the mist analytically
        /// alone (and the local volumes not at all).
        pub distance: f32,
        /// The share of light the mist scatters, by channel; the rest it absorbs.
        pub albedo: Vector,
        /// The Henyey-Greenstein asymmetry g (-0.95 to 0.95) of the mist's phase function:
        /// positive scatters toward where the light is going (shafts toward the sun), negative
        /// back toward it, 0 evenly.
        pub anisotropy: f32,
        /// How much of the sun's light the mist scatters, a scale on its in-scattering (0 for
        /// none).
        pub sun: f32,
        /// How much of the sky's and the probes' ambient light the mist scatters, a scale on
        /// its in-scattering.
        pub ambient: f32,
        /// How far the mist's ambient colour is the sky's own color in the direction looked,
        /// rather than the sky's mean light all round (0 to 1): 1 makes distant objects fade
        /// into the sky behind them.
        pub sky: f32,
        /// How much a 3D noise varies the global mist's density (0 to 1: 0 uniform, 1 from
        /// nothing to twice the mean); the mean density stays what it was.
        pub noise: f32,
        /// The size in world units of the noise's coarsest cell; finer octaves follow.
        pub noise_scale: f32,
        /// How far the game has carried the noise along x, y and z (wind times time): the same
        /// drift draws the same mist.
        pub drift: Vector,
    }
    impl Default for Fog {
        fn default() -> Self {
            Self {
                radius: Default::default(),
                density: Default::default(),
                height: 1.0,
                edge: 0.0,
                distance: 0.0,
                albedo: Vector { x: 1.0, y: 1.0, z: 1.0 },
                anisotropy: 0.0,
                sun: 1.0,
                ambient: 1.0,
                sky: 0.0,
                noise: 0.0,
                noise_scale: 16.0,
                drift: Vector { x: 0.0, y: 0.0, z: 0.0 },
            }
        }
    }
    /// The shadow map's view; `texels` 0 for no shadows.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Shadow {
        pub view: Matrix,
        /// The width of each sun cascade's map in texels; 0 for no sun shadows.
        pub texels: u32,
        /// How soft the sun's shadows are: 0 hard (one hardware-filtered tap), 1
        /// percentage-closer filtering over a rotated Poisson disk of a few texels, 2
        /// percentage-closer soft shadows (a blocker search sets the penumbra from `softness`,
        /// contact hardening).
        pub filter: u32,
        /// The sun's size for soft shadows: the width of the penumbra per unit of distance
        /// between the receiver and its blocker, about twice the tangent of the sun's angular
        /// radius. 0 keeps only the filter's own width.
        pub softness: f32,
        /// Steps of the screen-space contact shadow march toward the sun along the depth
        /// buffer; 0 for none. Needs a perspective camera, and draws the casters a second time
        /// as depth.
        pub contact: u32,
        /// How far toward the sun a contact shadow looks, in world units.
        pub contact_length: f32,
        /// The most point and spot lights that cast shadows this frame, picked by screen size
        /// and then by id (a point light costs six map faces, a spot one; at most 18 faces are
        /// drawn); 0 for none.
        pub local: u32,
        /// The width of each local light's map face in texels.
        pub local_texels: u32,
    }
    /// A lattice of diffuse light probes the game baked: where it is and how big, and the
    /// revision of the probes the host holds.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct ProbeVolume {
        /// Which bake the probes are: the host keeps the probes it was last sent, and uses them
        /// while this matches theirs.
        pub revision: u32,
        /// Probes along x; 0 for no probe volume.
        pub nx: u32,
        pub ny: u32,
        pub nz: u32,
        /// The distance between neighbouring probes, in world units.
        pub spacing: f32,
        /// Where probe 0 is; probe (z * ny + y) * nx + x lies at origin + (x, y, z) * spacing.
        pub origin: Vector,
    }
    /// Clouds over the scene: how much of the sky they cover, 0 for none, and how far the wind
    /// has carried them along x and z.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Clouds {
        pub cover: f32,
        pub drift: Drift,
    }
    /// How the host keeps local reflection probes up to date: how many cube faces it may draw
    /// in one frame.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Reflection {
        /// The most faces of reflection probe cubes drawn in one frame (6 make a whole probe);
        /// a probe whose revision changed, or that asks, is drawn again over as many frames as
        /// that takes. 0 draws none: the probes keep what they held.
        pub budget: u32,
    }
    /// How the camera exposes the frame: a fixed exposure in stops, or auto exposure that
    /// meters the scene on the GPU and adapts over time, with the limits and the speeds of the
    /// adaptation.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Exposure {
        /// Meter the scene on the GPU and adapt toward it (eye adaptation); off, `compensation`
        /// is a fixed exposure in stops.
        pub auto: bool,
        /// Stops of exposure added to the scene's light: the fixed exposure when `auto` is off,
        /// else a bias on what the metering finds. Within -16..16.
        pub compensation: f32,
        /// The least exposure auto exposure may adapt to, in stops. Within -16..16.
        pub min: f32,
        /// The most exposure auto exposure may adapt to, in stops. Within -16..16 and at least
        /// `min`.
        pub max: f32,
        /// How fast auto exposure rises when the scene darkens, as a rate per second: each
        /// frame closes 1 - exp(-rate * dt) of the distance to its target, so frame rate does
        /// not change the curve. 0 holds.
        pub brighten: f32,
        /// How fast auto exposure falls when the scene brightens, as a rate per second.
        pub darken: f32,
        /// The metering ignores pixels below this share of the picture's weight, darkest first
        /// (0 to 1).
        pub low: f32,
        /// The metering ignores pixels above this share of the picture's weight (0 to 1, above
        /// `low`).
        pub high: f32,
        /// How much the middle of the picture counts for more than its edges in the metering: 0
        /// weighs every pixel equally, 1 by a Gaussian of the distance from the centre (0 to
        /// 1).
        pub centre: f32,
        /// The darkest luminance the metering histogram tells apart, as log2 of linear
        /// luminance (1 is SDR white); darker pixels fall in its first bin.
        pub range_low: f32,
        /// The brightest luminance the histogram tells apart, as log2 of linear luminance,
        /// above `range_low` by at most 32; brighter pixels fall in its last bin.
        pub range_high: f32,
        /// The linear luminance the metered average is brought to (middle grey).
        pub target: f32,
        /// Seconds of the game's time since the last frame: auto exposure adapts by this much.
        /// 0 holds it, as a paused game should.
        pub dt: f32,
        /// The camera cut or the game restored an earlier moment: auto exposure jumps to its
        /// target this frame instead of adapting.
        pub cut: bool,
    }
    impl Default for Exposure {
        fn default() -> Self {
            Self {
                auto: Default::default(),
                compensation: 0.0,
                min: -6.0,
                max: 6.0,
                brighten: 3.0,
                darken: 1.0,
                low: 0.1,
                high: 0.9,
                centre: 0.0,
                range_low: -10.0,
                range_high: 6.0,
                target: 0.18,
                dt: 0.0,
                cut: Default::default(),
            }
        }
    }
    /// The color grade the present pass applies after exposure and before tone mapping: white
    /// balance, contrast about a pivot, ASC CDL lift, gamma and gain, saturation and vibrance,
    /// up to two blended look-up tables, and a vignette. The defaults grade nothing.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Grade {
        /// White balance along blue to orange, -1 cooler to 1 warmer, which scales red up and
        /// blue down by up to half a stop each.
        pub temperature: f32,
        /// White balance along green to magenta, -1 greener to 1 more magenta.
        pub tint: f32,
        /// Stops added to the light before grading (0 for none). Within -8..8.
        pub exposure: f32,
        /// Contrast about `pivot` in a log encoding of the light: 1 keeps it, 0 flattens, above
        /// 1 steepens (0 to 4).
        pub contrast: f32,
        /// The linear luminance contrast keeps in place (middle grey, 0.18).
        pub pivot: f32,
        /// How far colors lie from grey of the same luminance: 1 keeps it, 0 is grey (0 to 4).
        pub saturation: f32,
        /// Saturation that favours the duller colors: positive boosts them most, negative mutes
        /// the vivid (-1 to 1).
        pub vibrance: f32,
        /// ASC CDL offset per channel, added after the gain, in linear light (-0.5 to 0.5);
        /// lifts the shadows.
        pub lift: Vector,
        /// Per channel, the ASC CDL power is its reciprocal (0.1 to 4); above 1 brightens the
        /// midtones.
        pub gamma: Vector,
        /// ASC CDL slope per channel, scaling the light (0 to 8); moves the highlights.
        pub gain: Vector,
        /// A look-up table to grade with: 1 plus the catalog index of a material whose texture
        /// is its strip: N slices of N by N texels (N from 2 to 64, the strip N*N wide and N
        /// tall) side by side, blue picking the slice, red across it and green down, holding
        /// colors in the grade's ACEScc-like log encoding (pen-render's ARCHITECTURE.md); 0 for
        /// none.
        pub lut_a: u32,
        /// A second table, 1 plus its material's catalog index, to blend toward from `lut_a` (a
        /// zone transition); 0 for none.
        pub lut_b: u32,
        /// How far from `lut_a` toward `lut_b` the tables are blended, 0 to 1.
        pub lut_blend: f32,
        /// How much of the table grades the picture, 0 (none) to 1 (all of it).
        pub lut_weight: f32,
        /// How much the corners darken, 0 for none to 1 for black.
        pub vignette: f32,
        /// Where the vignette starts, as a share of the way from the middle to a corner (0 to
        /// 1).
        pub vignette_radius: f32,
        /// Display tonemapping operator: 0 Shoulder (default), 1 AgX filmic, 2 ACES Filmic, 3
        /// Khronos PBR Neutral.
        pub tonemap: u32,
        /// AMD FidelityFX RCAS contrast-adaptive sharpening strength in 0..1 (0 disables).
        pub sharpness: f32,
        /// Luminance-aware film grain intensity in 0..1 (0 disables).
        pub grain: f32,
        /// Horizontal chromatic anamorphic lens flare strength in 0..2 (0 disables).
        pub flare: f32,
        /// Depth/normal edge outline strength in 0..1 (0 disables).
        pub outline_strength: f32,
    }
    impl Default for Grade {
        fn default() -> Self {
            Self {
                temperature: 0.0,
                tint: 0.0,
                exposure: 0.0,
                contrast: 1.0,
                pivot: 0.18,
                saturation: 1.0,
                vibrance: 0.0,
                lift: Vector { x: 0.0, y: 0.0, z: 0.0 },
                gamma: Vector { x: 1.0, y: 1.0, z: 1.0 },
                gain: Vector { x: 1.0, y: 1.0, z: 1.0 },
                lut_a: Default::default(),
                lut_b: Default::default(),
                lut_blend: 0.0,
                lut_weight: 1.0,
                vignette: 0.0,
                vignette_radius: 0.5,
                tonemap: 0,
                sharpness: 0.0,
                grain: 0.0,
                flare: 0.0,
                outline_strength: 0.0,
            }
        }
    }
    /// Camera and per-object velocity motion blur applied in linear HDR light after TAA and
    /// depth of field.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct MotionBlur {
        /// Shutter scale on per-frame screen-space velocity for camera and per-object motion
        /// blur (0 disables the pass; 0.5 is a 180-degree shutter; within 0..4).
        pub strength: f32,
        /// Maximum velocity motion blur radius as a fraction of screen dimensions (within
        /// 0..0.25).
        pub max_radius: f32,
        /// Number of samples along the velocity vector for motion blur (clamped to 4..32 when
        /// active).
        pub samples: u32,
    }
    impl Default for MotionBlur {
        fn default() -> Self {
            Self {
                strength: 0.0,
                max_radius: 0.05,
                samples: 8,
            }
        }
    }
    /// Bokeh depth of field applied in linear HDR light after TAA and before motion blur and
    /// bloom.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Dof {
        /// Distance from the camera to the focal plane in world units (0 disables depth of
        /// field).
        pub focus_distance: f32,
        /// Depth band centered on `focus_distance` that remains fully sharp, in world units.
        pub focus_range: f32,
        /// Circle-of-confusion blur scale in screen-height UV units (0 disables depth of field;
        /// within 0..1).
        pub aperture: f32,
        /// Maximum bokeh disk radius as a fraction of screen height (within 0..0.1).
        pub max_blur: f32,
    }
    impl Default for Dof {
        fn default() -> Self {
            Self {
                focus_distance: 0.0,
                focus_range: 0.0,
                aperture: 0.0,
                max_blur: 0.025,
            }
        }
    }
    /// Shared reverb room RT60 decay and high-frequency damping; zeros use the default 1.5 s
    /// room.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct ReverbRoom {
        /// Shared reverb RT60 decay time in seconds; 0 uses the default 1.5 s room.
        pub decay: f32,
        /// Shared reverb high-frequency damping in 0..1; 0 uses the default curve, higher
        /// values darken the room faster.
        pub damping: f32,
    }
    impl Default for ReverbRoom {
        fn default() -> Self {
            Self {
                decay: 0.0,
                damping: 0.0,
            }
        }
    }
    /// One of the game's own sections: its tag, from its declaration, and its lanes.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct Extension {
        pub tag: u32,
        pub values: Vec<f32>,
    }
    /// The game's frame: what to draw and play, and what to do.
    #[derive(Clone, Debug, PartialEq)]
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
        /// filtering, ambient occlusion, a sky, screen effects, bloom and temporal
        /// antialiasing.
        pub quality: Quality,
        /// The sky's colors and the way toward the sun or moon.
        pub sky: Sky,
        /// A lattice of diffuse light probes the game baked: where it is and how big, and the
        /// revision of the probes the host holds.
        pub probe_volume: ProbeVolume,
        /// The participating medium: a mist beyond `radius` of the origin whose density falls
        /// off exponentially with height, drawn through a froxel volume out to `distance` (lit
        /// by the sun, the lights and the sky, with shafts and shadows) and analytically beyond
        /// it; density 0 and no fog volumes for none.
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
        /// The frame sprites are an application loading screen: draw them above the startup
        /// veil even while scene pipelines compile. Scene assets continue loading underneath.
        pub loading: bool,
        /// Show sound captions when the player has enabled them in the game's accessibility
        /// settings. Off by default; the game owns and saves the choice.
        pub show_captions: bool,
        /// Resolve persistent scene instances, inserting them before immediate draw
        /// retained_after.
        pub retained_enabled: bool,
        /// Replace the persistent instance scene. Required on its first frame.
        pub retained_reset: bool,
        pub retained_after: u32,
        /// How the camera exposes the frame: a fixed exposure in stops, or auto exposure that
        /// meters the scene on the GPU and adapts over time, with the limits and the speeds of
        /// the adaptation.
        pub exposure: Exposure,
        /// The color grade the present pass applies after exposure and before tone mapping:
        /// white balance, contrast about a pivot, ASC CDL lift, gamma and gain, saturation and
        /// vibrance, up to two blended look-up tables, and a vignette. The defaults grade
        /// nothing.
        pub grade: Grade,
        /// How the host keeps local reflection probes up to date: how many cube faces it may
        /// draw in one frame.
        pub reflection: Reflection,
        /// Camera and per-object velocity motion blur applied in linear HDR light after TAA and
        /// depth of field.
        pub motion_blur: MotionBlur,
        /// Bokeh depth of field applied in linear HDR light after TAA and before motion blur
        /// and bloom.
        pub dof: Dof,
        /// Shared reverb room RT60 decay and high-frequency damping; zeros use the default 1.5
        /// s room.
        pub reverb: ReverbRoom,
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
        /// The player's preferences to store, when they changed, as many as the game keeps; the
        /// host plays them back as Preference events when the game starts.
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
        /// The layers of skinned draws' poses: each samples one of its draw's mesh's clips at a
        /// time, and a draw's layers blend by their weights. A draw is posed when it has
        /// layers, its mesh a skin and its material's shader a `vertex_skinned` entry point;
        /// the host works out its joints' matrices and draws it alone.
        pub poses: Vec<Pose>,
        /// Particle emitters, which hosts simulate on the GPU from each emitter's own clock and
        /// seed: every particle's path follows from them, so a frame's particles depend on that
        /// frame's emitters alone.
        pub emitters: Vec<Emitter>,
        /// Point and spot lights beside the sun. The renderer sorts them into clusters of the
        /// view, which the shaders of lit materials read (pen-render's lights.wgsl).
        pub lights: Vec<Light>,
        /// The sun's shadow maps, nearest to the camera first. They share the light view
        /// `shadow.view` and differ in their orthographic windows; the game fits each round its
        /// slice of the camera's view and moves it in whole texels.
        pub cascades: Vec<Cascade>,
        /// The light probes of the volume `probe_volume` describes, in index order, sent when
        /// the game has a new bake; none keeps the probes the host holds (while their revision
        /// matches).
        pub probes: Vec<Probe>,
        /// Local reflection probes: boxes of the world whose glossy reflections the host draws
        /// from the probe's own point of view, as a cube, and blends by how far inside each box
        /// a surface is. The host captures a probe when its revision changes (or `capture`
        /// rises), within `reflection.budget` faces a frame, and keeps MAX_REFLECTION_SLOTS
        /// cubes, dropping the ones least worth keeping.
        pub reflection_probes: Vec<ReflectionProbe>,
        /// Local fog volumes: boxes and ellipsoids of the world, axis-aligned, whose medium
        /// adds to the global mist's inside them (when `fog.distance` is above 0 and the
        /// volumetrics tier is on): extinction `density` per world unit, scattering `albedo`,
        /// `emission`, and a falloff toward the edge.
        pub fog_volumes: Vec<FogVolume>,
        /// Where the panners are heard from, as WebAudio's AudioListener: none, the camera
        /// (`view`'s eye, facing where it looks, its up the view's), or one listener.
        pub listener: Vec<Listener>,
        /// Voices and sounds heard from somewhere in the world, as WebAudio's PannerNode
        /// (equal-power panning) places them relative to the listener; the rest play as they
        /// are.
        pub panners: Vec<Panner>,
        pub instances: Vec<Instance>,
        pub instance_removals: Vec<InstanceRemoval>,
        /// Explicit skinning joint matrices supplied by the game (e.g. from Ink's animation,
        /// IK, inertialization, ragdoll or secondary-bone solvers). Applied after any clip
        /// `poses` for the same draw, or starting from identity when no clip layers are
        /// declared.
        pub bone_matrices: Vec<BoneMatrix>,
        /// Blendshape / morph target weights applied to mesh vertices on the GPU before
        /// skinning and shading.
        pub morph_weights: Vec<MorphWeight>,
        /// Previous-frame model transforms for moving scene draws, used to generate per-object
        /// screen-space motion vectors for TAA reprojection and per-object motion blur.
        pub motions: Vec<Motion>,
        /// World-space camera-facing ribbon and beam trail control points rendered in the
        /// translucent pass.
        pub ribbons: Vec<RibbonPoint>,
        /// Dual-motor gamepad rumble / haptic commands for connected controllers.
        pub haptics: Vec<Haptic>,
        /// Spatial acoustic reverb zones and room portals blended smoothly around the listener.
        pub reverb_zones: Vec<ReverbZone>,
        /// Real-time procedural and granular DSP synthesizer voices mixed alongside clip
        /// voices.
        pub synths: Vec<SynthVoice>,
        /// Physically based LTC area lights (rect, disk, tube) and 2D cookie projectors.
        pub area_lights: Vec<AreaLightSpec>,
        /// Full PBR deferred decals projecting albedo, normal, ORM, and emissive onto opaque
        /// surfaces.
        pub pbr_decals: Vec<PbrDecalSpec>,
        /// Secondary cameras rendered either to dynamic material textures (portals, mirrors,
        /// monitors) or split-screen viewports.
        pub camera_targets: Vec<CameraTargetSpec>,
        /// GPU-decoded video streams bound to dynamic material textures.
        pub video_streams: Vec<VideoStreamSpec>,
        /// The game's own sections, which its pen.json `protocol` declares, after the rest of
        /// the packet: a marker, their count, then each one's tag, length and values. A packet
        /// without any ends before the marker. Tag 0 is reserved for one optional development
        /// provenance block: at most MAX_PROVENANCE_LANES UTF-8 JSON bytes as float32 lanes,
        /// version 1 with entries of first/end packet lane ranges and optional object, system,
        /// source (file/start/end UTF-8 byte span). See docs/packet-diagnostics.md. Hosts
        /// consume this metadata without retaining it in the rendered frame; release games omit
        /// it.
        pub extensions: Vec<Extension>,
    }
    impl Default for RenderFrame {
        fn default() -> Self {
            Self {
                save_requested: Default::default(),
                load_requested: Default::default(),
                scene_save_requested: Default::default(),
                scene_load_requested: Default::default(),
                checkpoint_requested: Default::default(),
                undo_requested: Default::default(),
                redo_requested: Default::default(),
                exit_requested: Default::default(),
                view: Default::default(),
                projection: Default::default(),
                clear: Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                ambient: Vector { x: 0.2, y: 0.2, z: 0.2 },
                light_direction: Vector { x: 0.0, y: -1.0, z: 0.0 },
                light_color: Vector { x: 1.0, y: 1.0, z: 1.0 },
                aberration: Default::default(),
                quality: Default::default(),
                sky: Default::default(),
                probe_volume: Default::default(),
                fog: Default::default(),
                shadow: Default::default(),
                blur: Default::default(),
                clouds: Default::default(),
                hide_cursor: Default::default(),
                loading: Default::default(),
                show_captions: Default::default(),
                retained_enabled: Default::default(),
                retained_reset: Default::default(),
                retained_after: Default::default(),
                exposure: Default::default(),
                grade: Default::default(),
                reflection: Default::default(),
                motion_blur: Default::default(),
                dof: Default::default(),
                reverb: Default::default(),
                draws: Default::default(),
                solids: Default::default(),
                decals: Default::default(),
                sounds: Default::default(),
                sprites: Default::default(),
                insets: Default::default(),
                preferences: Default::default(),
                fields: Default::default(),
                ui: Default::default(),
                services: Default::default(),
                buses: Default::default(),
                voices: Default::default(),
                poses: Default::default(),
                emitters: Default::default(),
                lights: Default::default(),
                cascades: Default::default(),
                probes: Default::default(),
                reflection_probes: Default::default(),
                fog_volumes: Default::default(),
                listener: Default::default(),
                panners: Default::default(),
                instances: Default::default(),
                instance_removals: Default::default(),
                bone_matrices: Default::default(),
                morph_weights: Default::default(),
                motions: Default::default(),
                ribbons: Default::default(),
                haptics: Default::default(),
                reverb_zones: Default::default(),
                synths: Default::default(),
                area_lights: Default::default(),
                pbr_decals: Default::default(),
                camera_targets: Default::default(),
                video_streams: Default::default(),
                extensions: Default::default(),
            }
        }
    }
}

/// The packets' layout: constants, the input packet encoder and the render
/// packet decoder; and the packets' canonical walks, which tests compare across
/// languages.
pub mod codec {
    use anyhow::{Result, bail, ensure};

    use super::types::*;

    /// Reserved extension tag for optional development provenance; game section tags start at
    /// 1.
    pub const PROVENANCE_TAG: usize = 0;
    /// The most UTF-8 JSON bytes, each encoded as one float32 lane, in a frame's provenance.
    pub const MAX_PROVENANCE_LANES: usize = 16384;
    /// The most source/object associations in one frame; no history is retained by the decoder.
    pub const MAX_PROVENANCE_ENTRIES: usize = 64;
    /// The most UTF-8 bytes in a provenance object ID, system label or source filename.
    pub const MAX_PROVENANCE_TEXT: usize = 256;
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
    pub const MAX_INSETS: usize = 64;
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
    /// The most explicit joint matrices one frame may declare.
    pub const MAX_BONE_MATRICES: usize = 65536;
    /// The most morph target weights one frame may declare.
    pub const MAX_MORPH_WEIGHTS: usize = 65536;
    /// The most per-draw previous transforms one frame may declare for motion vectors.
    pub const MAX_MOTIONS: usize = 100000;
    /// The most ribbon trail control points one frame may declare.
    pub const MAX_RIBBONS: usize = 65536;
    /// The most gamepad haptic rumble commands one frame may declare.
    pub const MAX_HAPTICS: usize = 32;
    /// The most LTC area lights one frame may declare.
    pub const MAX_AREA_LIGHTS: usize = 64;
    /// The most PBR deferred decals one frame may declare.
    pub const MAX_PBR_DECALS: usize = 1024;
    /// The most render-to-texture or split-screen secondary cameras one frame may declare.
    pub const MAX_CAMERA_TARGETS: usize = 8;
    /// The most GPU video streams one frame may declare.
    pub const MAX_VIDEO_STREAMS: usize = 16;
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
    /// The most pose layers one frame may have.
    pub const MAX_POSES: usize = 65536;
    /// The most particle emitters one frame may have.
    pub const MAX_EMITTERS: usize = 1024;
    /// The most particles one emitter has alive at once (its rate times its lifetime, rounded
    /// up); hosts leave out the rest.
    pub const MAX_PARTICLES: usize = 65536;
    /// The most of a game's own sections one frame may carry.
    pub const MAX_EXTENSIONS: usize = 64;
    /// A game's own sections and events are numbered from 1 to this.
    pub const MAX_EXTENSION_TAG: usize = 65535;
    /// The most lanes one of a game's own sections may carry.
    pub const MAX_EXTENSION_LANES: usize = 1048576;
    /// Gamepads connected at once that the game hears of: pads are numbered from 0 to one less
    /// than this.
    pub const MAX_GAMEPADS: usize = 16;
    /// Actions a game declares (the input service's declare): they are numbered from 0 to one
    /// less than this.
    pub const MAX_ACTIONS: usize = 256;
    /// Bindings one action holds, its defaults and the player's alike.
    pub const MAX_ACTION_BINDINGS: usize = 8;
    /// Touches down at once that the game hears of: they are numbered from 0 to one less than
    /// this.
    pub const MAX_TOUCHES: usize = 32;
    /// The most point and spot lights one frame may have.
    pub const MAX_LIGHTS: usize = 4096;
    /// The most sun shadow maps one frame may have.
    pub const MAX_CASCADES: usize = 4;
    /// The most light probes one frame may carry (a volume may be larger than a frame's share
    /// only by sending none: see `probe_volume.revision`).
    pub const MAX_PROBES: usize = 16384;
    /// The most reflection probes one frame may publish; the host keeps the nearest and most
    /// important few in its cube array (MAX_REFLECTION_SLOTS).
    pub const MAX_REFLECTION_PROBES: usize = 64;
    /// The most local fog volumes one frame may publish.
    pub const MAX_FOG_VOLUMES: usize = 32;
    /// Preference events from this index on carry the engine's preferences
    /// (ENGINE_PREFERENCE_*); below it, the game's own.
    pub const ENGINE_PREFERENCES: usize = 64;
    /// The most panners one frame may declare: one for each voice and sound it may have.
    pub const MAX_PANNERS: usize = 512;
    /// The most spatial reverb zones and acoustic portals one frame may declare.
    pub const MAX_REVERB_ZONES: usize = 32;
    /// The most procedural or granular DSP synth voices one frame may declare.
    pub const MAX_SYNTHS: usize = 64;
    pub const MAX_INSTANCES: usize = 100000;
    pub const MAX_INSTANCE_REMOVALS: usize = 100000;
    pub const PACKET_MAGIC: f32 = 0x474446 as f32;
    /// The lane that starts a packet's block of the game's own sections.
    pub const EXTENSIONS_MARKER: f32 = 0x504558 as f32;
    /// The render packet header: magic, schema, state_count, draw_count, solid_count,
    /// solid_lanes, decal_count, sound_count, sprite_count, inset_count, inset_draw_count,
    /// preference_count, save_requested, load_requested, scene_save_requested,
    /// scene_load_requested, checkpoint_requested, undo_requested, redo_requested,
    /// exit_requested, view16, projection16, clear4, ambient3, light_direction3, light_color3,
    /// aberration2, quality.msaa, quality.fxaa, quality.scale, quality.filtering,
    /// quality.occlusion, quality.sky, quality.effects, sky.zenith3, sky.horizon3, sky.ground3,
    /// sky.sun3, sky.night, sky.tint3, sky.sh03, sky.sh13, sky.sh23, sky.sh33, sky.sh43,
    /// sky.sh53, sky.sh63, sky.sh73, sky.sh83, probe_volume.revision, probe_volume.nx,
    /// probe_volume.ny, probe_volume.nz, probe_volume.spacing, probe_volume.origin3,
    /// fog.radius, fog.density, fog.height, shadow.view16, shadow.texels, shadow.filter,
    /// shadow.softness, shadow.contact, shadow.contact_length, shadow.local,
    /// shadow.local_texels, blur, quality.bloom, clouds.cover, clouds.drift2, hide_cursor,
    /// field_count, field_lanes, ui_count, ui_lanes, service_count, service_lanes, bus_count,
    /// voice_count, pose_count, emitter_count, quality.taa, light_count, cascade_count,
    /// probe_count, reflection_probe_count, listener_count, panner_count, loading,
    /// show_captions, retained_enabled, retained_reset, retained_after, instance_count,
    /// instance_removal_count, exposure.auto, exposure.compensation, exposure.min,
    /// exposure.max, exposure.brighten, exposure.darken, exposure.low, exposure.high,
    /// exposure.centre, exposure.range_low, exposure.range_high, exposure.target, exposure.dt,
    /// exposure.cut, grade.temperature, grade.tint, grade.exposure, grade.contrast,
    /// grade.pivot, grade.saturation, grade.vibrance, grade.lift3, grade.gamma3, grade.gain3,
    /// grade.lut_a, grade.lut_b, grade.lut_blend, grade.lut_weight, grade.vignette,
    /// grade.vignette_radius, quality.ssr, reflection.budget, fog.edge, fog.distance,
    /// fog.albedo3, fog.anisotropy, fog.sun, fog.ambient, fog.sky, fog.noise, fog.noise_scale,
    /// fog.drift3, quality.volumetrics, fog_volume_count, bone_matrix_count,
    /// morph_weight_count, motion_blur.strength, motion_blur.max_radius, motion_blur.samples,
    /// dof.focus_distance, dof.focus_range, dof.aperture, dof.max_blur, motion_count,
    /// quality.ssgi, ribbon_count, reverb.decay, reverb.damping, haptic_count,
    /// reverb_zone_count, synth_count, grade.tonemap, grade.sharpness, grade.grain,
    /// grade.flare, grade.outline_strength, quality.oit, quality.physical_sky, quality.upscale,
    /// quality.ray_tracing, quality.ddgi, area_light_count, pbr_decal_count,
    /// camera_target_count, video_stream_count.
    pub const PACKET_HEADER_FLOATS: usize = 266;
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
    /// light_color3, draws count, pixel_size, after_sprites.
    pub const INSET_FLOATS: usize = 48;
    /// Lanes: preferences lanes.
    pub const PREFERENCE_FLOATS: usize = 256;
    /// Lanes: material, width, height, x, z, columns, rows, before a field's values.
    pub const FIELD_FLOATS: usize = 7;
    /// Lanes: id, parent, role, x, y, width, height, focusable, focused, disabled, hidden,
    /// modal, checked, expanded, selected, live, value, minimum, maximum, step, actions,
    /// text_length, name_length, description_length, before a ui's text.
    pub const UI_NODE_FLOATS: usize = 24;
    /// Lanes: request, service, operation, a, b, c, d, value_count, before a service's values.
    pub const SERVICE_REQUEST_FLOATS: usize = 8;
    /// Lanes: gain, lowpass, reverb, comp_threshold, comp_ratio, duck_bus, duck_amount,
    /// eq_low_db, eq_mid_db, eq_high_db.
    pub const BUS_FLOATS: usize = 10;
    /// Lanes: id, clip, bus, gain, pan, pitch, lowpass, reverb, looping, start, fade_in,
    /// fade_out.
    pub const VOICE_FLOATS: usize = 12;
    /// Lanes: draw, clip, time, weight, inset.
    pub const POSE_FLOATS: usize = 5;
    /// Lanes: material, transform16, seed, time, duration, rate, lifetime, radius, velocity3,
    /// spread, gravity3, drag, size_start, size_end, color_start4, color_end4, additive,
    /// soft_depth, lit, flipbook_cols, flipbook_rows, flipbook_fps.
    pub const EMITTER_FLOATS: usize = 47;
    /// Lanes: position3, radius, color3, direction3, inner, outer, shadow, id.
    pub const LIGHT_FLOATS: usize = 14;
    /// Lanes: projection16, texel, far.
    pub const CASCADE_FLOATS: usize = 18;
    /// Lanes: sh03, sh13, sh23, sh33, sh43, sh53, sh63, sh73, sh83, valid.
    pub const PROBE_FLOATS: usize = 28;
    /// Lanes: id, revision, priority, capture, position3, center3, extent3, blend,
    /// parallax_center3, parallax_extent3, far.
    pub const REFLECTION_PROBE_FLOATS: usize = 21;
    /// Lanes: shape, center3, extent3, density, albedo3, emission3, falloff.
    pub const FOG_VOLUME_FLOATS: usize = 15;
    /// Lanes: position3, forward3, up3, velocity3.
    pub const LISTENER_FLOATS: usize = 12;
    /// Lanes: voice, sound, position3, orientation3, distance_model, ref_distance,
    /// max_distance, rolloff, cone_inner, cone_outer, cone_outer_gain, hrtf, velocity3,
    /// doppler, occlusion, zone.
    pub const PANNER_FLOATS: usize = 22;
    /// Lanes: id_low, id_high, order, mesh, material, transform16, parameters count and
    /// parameters4.
    pub const INSTANCE_FLOATS: usize = 26;
    /// Lanes: id_low, id_high.
    pub const INSTANCE_REMOVAL_FLOATS: usize = 2;
    /// Lanes: inset, draw, joint, matrix16.
    pub const BONE_MATRIX_FLOATS: usize = 19;
    /// Lanes: inset, draw, target, weight.
    pub const MORPH_WEIGHT_FLOATS: usize = 4;
    /// Lanes: draw, previous16.
    pub const MOTION_FLOATS: usize = 17;
    /// Lanes: ribbon, material, additive, position3, width, color4, soft_depth.
    pub const RIBBON_FLOATS: usize = 12;
    /// Lanes: pad, low, high, duration.
    pub const HAPTIC_FLOATS: usize = 4;
    /// Lanes: id, position3, half_extents3, blend, decay, damping, wet, portal_attenuation.
    pub const REVERB_ZONE_FLOATS: usize = 12;
    /// Lanes: id, mode, bus, frequency, gain, pan, detune, attack, decay, sustain, release,
    /// cutoff, resonance, grain_size, grain_pitch, reverb.
    pub const SYNTH_VOICE_FLOATS: usize = 16;
    /// Lanes: kind, position3, right3, up3, color3, intensity, width, height, range,
    /// cookie_material, two_sided.
    pub const AREA_LIGHT_FLOATS: usize = 19;
    /// Lanes: material, transform16, albedo4, normal_strength, roughness, metallic,
    /// emissive_strength, fade.
    pub const PBR_DECAL_FLOATS: usize = 26;
    /// Lanes: material, width, height, viewport_x, viewport_y, viewport_w, viewport_h,
    /// split_screen, view16, projection16.
    pub const CAMERA_TARGET_FLOATS: usize = 40;
    /// Lanes: material, time, playing, looping, speed.
    pub const VIDEO_STREAM_FLOATS: usize = 5;
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
            pub const SKY_TINT: usize = 87;
            pub const SKY_SH0: usize = 90;
            pub const SKY_SH1: usize = 93;
            pub const SKY_SH2: usize = 96;
            pub const SKY_SH3: usize = 99;
            pub const SKY_SH4: usize = 102;
            pub const SKY_SH5: usize = 105;
            pub const SKY_SH6: usize = 108;
            pub const SKY_SH7: usize = 111;
            pub const SKY_SH8: usize = 114;
            pub const PROBE_VOLUME_REVISION: usize = 117;
            pub const PROBE_VOLUME_NX: usize = 118;
            pub const PROBE_VOLUME_NY: usize = 119;
            pub const PROBE_VOLUME_NZ: usize = 120;
            pub const PROBE_VOLUME_SPACING: usize = 121;
            pub const PROBE_VOLUME_ORIGIN: usize = 122;
            pub const FOG_RADIUS: usize = 125;
            pub const FOG_DENSITY: usize = 126;
            pub const FOG_HEIGHT: usize = 127;
            pub const SHADOW_VIEW: usize = 128;
            pub const SHADOW_TEXELS: usize = 144;
            pub const SHADOW_FILTER: usize = 145;
            pub const SHADOW_SOFTNESS: usize = 146;
            pub const SHADOW_CONTACT: usize = 147;
            pub const SHADOW_CONTACT_LENGTH: usize = 148;
            pub const SHADOW_LOCAL: usize = 149;
            pub const SHADOW_LOCAL_TEXELS: usize = 150;
            pub const BLUR: usize = 151;
            pub const QUALITY_BLOOM: usize = 152;
            pub const CLOUDS_COVER: usize = 153;
            pub const CLOUDS_DRIFT: usize = 154;
            pub const HIDE_CURSOR: usize = 156;
            pub const FIELD_COUNT: usize = 157;
            pub const FIELD_LANES: usize = 158;
            pub const UI_COUNT: usize = 159;
            pub const UI_LANES: usize = 160;
            pub const SERVICE_COUNT: usize = 161;
            pub const SERVICE_LANES: usize = 162;
            pub const BUS_COUNT: usize = 163;
            pub const VOICE_COUNT: usize = 164;
            pub const POSE_COUNT: usize = 165;
            pub const EMITTER_COUNT: usize = 166;
            pub const QUALITY_TAA: usize = 167;
            pub const LIGHT_COUNT: usize = 168;
            pub const CASCADE_COUNT: usize = 169;
            pub const PROBE_COUNT: usize = 170;
            pub const REFLECTION_PROBE_COUNT: usize = 171;
            pub const LISTENER_COUNT: usize = 172;
            pub const PANNER_COUNT: usize = 173;
            pub const LOADING: usize = 174;
            pub const SHOW_CAPTIONS: usize = 175;
            pub const RETAINED_ENABLED: usize = 176;
            pub const RETAINED_RESET: usize = 177;
            pub const RETAINED_AFTER: usize = 178;
            pub const INSTANCE_COUNT: usize = 179;
            pub const INSTANCE_REMOVAL_COUNT: usize = 180;
            pub const EXPOSURE_AUTO: usize = 181;
            pub const EXPOSURE_COMPENSATION: usize = 182;
            pub const EXPOSURE_MIN: usize = 183;
            pub const EXPOSURE_MAX: usize = 184;
            pub const EXPOSURE_BRIGHTEN: usize = 185;
            pub const EXPOSURE_DARKEN: usize = 186;
            pub const EXPOSURE_LOW: usize = 187;
            pub const EXPOSURE_HIGH: usize = 188;
            pub const EXPOSURE_CENTRE: usize = 189;
            pub const EXPOSURE_RANGE_LOW: usize = 190;
            pub const EXPOSURE_RANGE_HIGH: usize = 191;
            pub const EXPOSURE_TARGET: usize = 192;
            pub const EXPOSURE_DT: usize = 193;
            pub const EXPOSURE_CUT: usize = 194;
            pub const GRADE_TEMPERATURE: usize = 195;
            pub const GRADE_TINT: usize = 196;
            pub const GRADE_EXPOSURE: usize = 197;
            pub const GRADE_CONTRAST: usize = 198;
            pub const GRADE_PIVOT: usize = 199;
            pub const GRADE_SATURATION: usize = 200;
            pub const GRADE_VIBRANCE: usize = 201;
            pub const GRADE_LIFT: usize = 202;
            pub const GRADE_GAMMA: usize = 205;
            pub const GRADE_GAIN: usize = 208;
            pub const GRADE_LUT_A: usize = 211;
            pub const GRADE_LUT_B: usize = 212;
            pub const GRADE_LUT_BLEND: usize = 213;
            pub const GRADE_LUT_WEIGHT: usize = 214;
            pub const GRADE_VIGNETTE: usize = 215;
            pub const GRADE_VIGNETTE_RADIUS: usize = 216;
            pub const QUALITY_SSR: usize = 217;
            pub const REFLECTION_BUDGET: usize = 218;
            pub const FOG_EDGE: usize = 219;
            pub const FOG_DISTANCE: usize = 220;
            pub const FOG_ALBEDO: usize = 221;
            pub const FOG_ANISOTROPY: usize = 224;
            pub const FOG_SUN: usize = 225;
            pub const FOG_AMBIENT: usize = 226;
            pub const FOG_SKY: usize = 227;
            pub const FOG_NOISE: usize = 228;
            pub const FOG_NOISE_SCALE: usize = 229;
            pub const FOG_DRIFT: usize = 230;
            pub const QUALITY_VOLUMETRICS: usize = 233;
            pub const FOG_VOLUME_COUNT: usize = 234;
            pub const BONE_MATRIX_COUNT: usize = 235;
            pub const MORPH_WEIGHT_COUNT: usize = 236;
            pub const MOTION_BLUR_STRENGTH: usize = 237;
            pub const MOTION_BLUR_MAX_RADIUS: usize = 238;
            pub const MOTION_BLUR_SAMPLES: usize = 239;
            pub const DOF_FOCUS_DISTANCE: usize = 240;
            pub const DOF_FOCUS_RANGE: usize = 241;
            pub const DOF_APERTURE: usize = 242;
            pub const DOF_MAX_BLUR: usize = 243;
            pub const MOTION_COUNT: usize = 244;
            pub const QUALITY_SSGI: usize = 245;
            pub const RIBBON_COUNT: usize = 246;
            pub const REVERB_DECAY: usize = 247;
            pub const REVERB_DAMPING: usize = 248;
            pub const HAPTIC_COUNT: usize = 249;
            pub const REVERB_ZONE_COUNT: usize = 250;
            pub const SYNTH_COUNT: usize = 251;
            pub const GRADE_TONEMAP: usize = 252;
            pub const GRADE_SHARPNESS: usize = 253;
            pub const GRADE_GRAIN: usize = 254;
            pub const GRADE_FLARE: usize = 255;
            pub const GRADE_OUTLINE_STRENGTH: usize = 256;
            pub const QUALITY_OIT: usize = 257;
            pub const QUALITY_PHYSICAL_SKY: usize = 258;
            pub const QUALITY_UPSCALE: usize = 259;
            pub const QUALITY_RAY_TRACING: usize = 260;
            pub const QUALITY_DDGI: usize = 261;
            pub const AREA_LIGHT_COUNT: usize = 262;
            pub const PBR_DECAL_COUNT: usize = 263;
            pub const CAMERA_TARGET_COUNT: usize = 264;
            pub const VIDEO_STREAM_COUNT: usize = 265;
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
            pub const AFTER_SPRITES: usize = 47;
        }
        pub mod bus {
            pub const GAIN: usize = 0;
            pub const LOWPASS: usize = 1;
            pub const REVERB: usize = 2;
            pub const COMP_THRESHOLD: usize = 3;
            pub const COMP_RATIO: usize = 4;
            pub const DUCK_BUS: usize = 5;
            pub const DUCK_AMOUNT: usize = 6;
            pub const EQ_LOW_DB: usize = 7;
            pub const EQ_MID_DB: usize = 8;
            pub const EQ_HIGH_DB: usize = 9;
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
        pub mod pose {
            pub const DRAW: usize = 0;
            pub const CLIP: usize = 1;
            pub const TIME: usize = 2;
            pub const WEIGHT: usize = 3;
            pub const INSET: usize = 4;
        }
        pub mod emitter {
            pub const MATERIAL: usize = 0;
            pub const TRANSFORM: usize = 1;
            pub const SEED: usize = 17;
            pub const TIME: usize = 18;
            pub const DURATION: usize = 19;
            pub const RATE: usize = 20;
            pub const LIFETIME: usize = 21;
            pub const RADIUS: usize = 22;
            pub const VELOCITY: usize = 23;
            pub const SPREAD: usize = 26;
            pub const GRAVITY: usize = 27;
            pub const DRAG: usize = 30;
            pub const SIZE_START: usize = 31;
            pub const SIZE_END: usize = 32;
            pub const COLOR_START: usize = 33;
            pub const COLOR_END: usize = 37;
            pub const ADDITIVE: usize = 41;
            pub const SOFT_DEPTH: usize = 42;
            pub const LIT: usize = 43;
            pub const FLIPBOOK_COLS: usize = 44;
            pub const FLIPBOOK_ROWS: usize = 45;
            pub const FLIPBOOK_FPS: usize = 46;
        }
        pub mod light {
            pub const POSITION: usize = 0;
            pub const RADIUS: usize = 3;
            pub const COLOR: usize = 4;
            pub const DIRECTION: usize = 7;
            pub const INNER: usize = 10;
            pub const OUTER: usize = 11;
            pub const SHADOW: usize = 12;
            pub const ID: usize = 13;
        }
        pub mod cascade {
            pub const PROJECTION: usize = 0;
            pub const TEXEL: usize = 16;
            pub const FAR: usize = 17;
        }
        pub mod probe {
            pub const SH0: usize = 0;
            pub const SH1: usize = 3;
            pub const SH2: usize = 6;
            pub const SH3: usize = 9;
            pub const SH4: usize = 12;
            pub const SH5: usize = 15;
            pub const SH6: usize = 18;
            pub const SH7: usize = 21;
            pub const SH8: usize = 24;
            pub const VALID: usize = 27;
        }
        pub mod reflection_probe {
            pub const ID: usize = 0;
            pub const REVISION: usize = 1;
            pub const PRIORITY: usize = 2;
            pub const CAPTURE: usize = 3;
            pub const POSITION: usize = 4;
            pub const CENTER: usize = 7;
            pub const EXTENT: usize = 10;
            pub const BLEND: usize = 13;
            pub const PARALLAX_CENTER: usize = 14;
            pub const PARALLAX_EXTENT: usize = 17;
            pub const FAR: usize = 20;
        }
        pub mod fog_volume {
            pub const SHAPE: usize = 0;
            pub const CENTER: usize = 1;
            pub const EXTENT: usize = 4;
            pub const DENSITY: usize = 7;
            pub const ALBEDO: usize = 8;
            pub const EMISSION: usize = 11;
            pub const FALLOFF: usize = 14;
        }
        pub mod listener {
            pub const POSITION: usize = 0;
            pub const FORWARD: usize = 3;
            pub const UP: usize = 6;
            pub const VELOCITY: usize = 9;
        }
        pub mod panner {
            pub const VOICE: usize = 0;
            pub const SOUND: usize = 1;
            pub const POSITION: usize = 2;
            pub const ORIENTATION: usize = 5;
            pub const DISTANCE_MODEL: usize = 8;
            pub const REF_DISTANCE: usize = 9;
            pub const MAX_DISTANCE: usize = 10;
            pub const ROLLOFF: usize = 11;
            pub const CONE_INNER: usize = 12;
            pub const CONE_OUTER: usize = 13;
            pub const CONE_OUTER_GAIN: usize = 14;
            pub const HRTF: usize = 15;
            pub const VELOCITY: usize = 16;
            pub const DOPPLER: usize = 19;
            pub const OCCLUSION: usize = 20;
            pub const ZONE: usize = 21;
        }
        pub mod instance {
            pub const ID_LOW: usize = 0;
            pub const ID_HIGH: usize = 1;
            pub const ORDER: usize = 2;
            pub const MESH: usize = 3;
            pub const MATERIAL: usize = 4;
            pub const TRANSFORM: usize = 5;
            pub const PARAMETERS: usize = 21;
        }
        pub mod instance_removal {
            pub const ID_LOW: usize = 0;
            pub const ID_HIGH: usize = 1;
        }
        pub mod bone_matrix {
            pub const INSET: usize = 0;
            pub const DRAW: usize = 1;
            pub const JOINT: usize = 2;
            pub const MATRIX: usize = 3;
        }
        pub mod morph_weight {
            pub const INSET: usize = 0;
            pub const DRAW: usize = 1;
            pub const TARGET: usize = 2;
            pub const WEIGHT: usize = 3;
        }
        pub mod motion {
            pub const DRAW: usize = 0;
            pub const PREVIOUS: usize = 1;
        }
        pub mod ribbon_point {
            pub const RIBBON: usize = 0;
            pub const MATERIAL: usize = 1;
            pub const ADDITIVE: usize = 2;
            pub const POSITION: usize = 3;
            pub const WIDTH: usize = 6;
            pub const COLOR: usize = 7;
            pub const SOFT_DEPTH: usize = 11;
        }
        pub mod haptic {
            pub const PAD: usize = 0;
            pub const LOW: usize = 1;
            pub const HIGH: usize = 2;
            pub const DURATION: usize = 3;
        }
        pub mod reverb_zone {
            pub const ID: usize = 0;
            pub const POSITION: usize = 1;
            pub const HALF_EXTENTS: usize = 4;
            pub const BLEND: usize = 7;
            pub const DECAY: usize = 8;
            pub const DAMPING: usize = 9;
            pub const WET: usize = 10;
            pub const PORTAL_ATTENUATION: usize = 11;
        }
        pub mod synth_voice {
            pub const ID: usize = 0;
            pub const MODE: usize = 1;
            pub const BUS: usize = 2;
            pub const FREQUENCY: usize = 3;
            pub const GAIN: usize = 4;
            pub const PAN: usize = 5;
            pub const DETUNE: usize = 6;
            pub const ATTACK: usize = 7;
            pub const DECAY: usize = 8;
            pub const SUSTAIN: usize = 9;
            pub const RELEASE: usize = 10;
            pub const CUTOFF: usize = 11;
            pub const RESONANCE: usize = 12;
            pub const GRAIN_SIZE: usize = 13;
            pub const GRAIN_PITCH: usize = 14;
            pub const REVERB: usize = 15;
        }
        pub mod area_light_spec {
            pub const KIND: usize = 0;
            pub const POSITION: usize = 1;
            pub const RIGHT: usize = 4;
            pub const UP: usize = 7;
            pub const COLOR: usize = 10;
            pub const INTENSITY: usize = 13;
            pub const WIDTH: usize = 14;
            pub const HEIGHT: usize = 15;
            pub const RANGE: usize = 16;
            pub const COOKIE_MATERIAL: usize = 17;
            pub const TWO_SIDED: usize = 18;
        }
        pub mod pbr_decal_spec {
            pub const MATERIAL: usize = 0;
            pub const TRANSFORM: usize = 1;
            pub const ALBEDO: usize = 17;
            pub const NORMAL_STRENGTH: usize = 21;
            pub const ROUGHNESS: usize = 22;
            pub const METALLIC: usize = 23;
            pub const EMISSIVE_STRENGTH: usize = 24;
            pub const FADE: usize = 25;
        }
        pub mod camera_target_spec {
            pub const MATERIAL: usize = 0;
            pub const WIDTH: usize = 1;
            pub const HEIGHT: usize = 2;
            pub const VIEWPORT_X: usize = 3;
            pub const VIEWPORT_Y: usize = 4;
            pub const VIEWPORT_W: usize = 5;
            pub const VIEWPORT_H: usize = 6;
            pub const SPLIT_SCREEN: usize = 7;
            pub const VIEW: usize = 8;
            pub const PROJECTION: usize = 24;
        }
        pub mod video_stream_spec {
            pub const MATERIAL: usize = 0;
            pub const TIME: usize = 1;
            pub const PLAYING: usize = 2;
            pub const LOOPING: usize = 3;
            pub const SPEED: usize = 4;
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
        pub const GAMEPAD_CONNECTED: u32 = 15;
        pub const GAMEPAD_DISCONNECTED: u32 = 16;
        pub const GAMEPAD_BUTTON: u32 = 17;
        pub const GAMEPAD_AXIS: u32 = 18;
        pub const ACTION: u32 = 19;
        pub const TEXT_INPUT: u32 = 20;
        pub const COMPOSITION: u32 = 21;
        pub const MOUSE_MOTION: u32 = 22;
        pub const TOUCH: u32 = 23;
        pub const PAN: u32 = 24;
        pub const PINCH: u32 = 25;
        pub const PLAYER: u32 = 26;
        pub const SESSION_STARTED: u32 = 27;
        pub const RENDERER_READY: u32 = 28;
        pub const GAME: u32 = 100;
    }

    #[inline]
    fn all_finite(values: &[f32]) -> bool {
        const EXP_MASK: u32 = 0x7f80_0000;
        let (chunks, rest) = values.as_chunks::<16>();
        for chunk in chunks {
            let mut nonfinite = 0u32;
            for &value in chunk {
                nonfinite |= u32::from((value.to_bits() & EXP_MASK) == EXP_MASK);
            }
            if nonfinite != 0 {
                return false;
            }
        }
        rest.iter().all(|value| value.is_finite())
    }

    pub fn validate_state(state: &[f32]) -> Result<()> {
        ensure!(
            state.len() <= MAX_STATE_FLOATS && all_finite(state),
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
            "Insert" => 0x11010d,
            "F1" => 0x110201,
            "F2" => 0x110202,
            "F3" => 0x110203,
            "F4" => 0x110204,
            "F5" => 0x110205,
            "F6" => 0x110206,
            "F7" => 0x110207,
            "F8" => 0x110208,
            "F9" => 0x110209,
            "F10" => 0x11020a,
            "F11" => 0x11020b,
            "F12" => 0x11020c,
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
            GameEvent::GamepadConnected { pad, standard } => [15.0, *pad as f32, *standard as f32, 0.0, 0.0],
            GameEvent::GamepadDisconnected { pad } => [16.0, *pad as f32, 0.0, 0.0, 0.0],
            GameEvent::GamepadButton { button, pad, value, pressed } => [17.0, *button as f32, *pad as f32, *value, *pressed as f32],
            GameEvent::GamepadAxis { axis, pad, value } => [18.0, *axis as f32, *pad as f32, *value, 0.0],
            GameEvent::Action { action, value, device } => [19.0, *action as f32, *value, *device as f32, 0.0],
            GameEvent::TextInput { character, composing } => [20.0, *character as f32, *composing as f32, 0.0, 0.0],
            GameEvent::Composition { length, cursor } => [21.0, *length as f32, *cursor as f32, 0.0, 0.0],
            GameEvent::MouseMotion { dx, dy } => [22.0, 0.0, *dx, *dy, 0.0],
            GameEvent::Touch { id, x, y, phase } => [23.0, *id as f32, *x, *y, *phase as f32],
            GameEvent::Pan { touches, dx, dy } => [24.0, *touches as f32, *dx, *dy, 0.0],
            GameEvent::Pinch { x, y, scale } => [25.0, 0.0, *x, *y, *scale],
            GameEvent::Player { index } => [26.0, *index as f32, 0.0, 0.0, 0.0],
            GameEvent::SessionStarted { restored } => [27.0, *restored as f32, 0.0, 0.0, 0.0],
            GameEvent::RendererReady => [28.0, 0.0, 0.0, 0.0, 0.0],
            GameEvent::Game { event, a, b, c } => [100.0, *event as f32, *a, *b, *c],
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
        ensure!(all_finite(&packet[..packet.len() - state.len()]), "frame contains a nonfinite float32 value");
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
            if !(0.0..=maximum as f32).contains(&value) || (value < 8_388_608.0 && (value as u32) as f32 != value) {
                bail!("game returned an invalid frame count or flag");
            }
            Ok(value as usize)
        }

        #[inline]
        fn flag(&self, index: usize) -> Result<bool> {
            let value = self.packet[index];
            if value == 0.0 {
                Ok(false)
            } else if value == 1.0 {
                Ok(true)
            } else {
                bail!("game returned an invalid frame count or flag");
            }
        }

        #[inline]
        fn matrix(&self, index: usize) -> Matrix {
            self.packet[index..index + 16].try_into().unwrap()
        }

        #[inline]
        fn mesh_at(&self, index: usize) -> Result<u32> {
            ensure!(!self.catalog.meshes.is_empty(), "game drew a mesh with no meshes in the catalog");
            Ok(self.count(index, self.catalog.meshes.len() - 1)? as u32)
        }

        #[inline]
        fn material_at(&self, index: usize) -> Result<u32> {
            ensure!(!self.catalog.materials.is_empty(), "game used a material with no materials in the catalog");
            Ok(self.count(index, self.catalog.materials.len() - 1)? as u32)
        }

        #[inline]
        fn sound_at(&self, index: usize) -> Result<u32> {
            ensure!(!self.catalog.sounds.is_empty(), "game played a sound with no sounds in the catalog");
            Ok(self.count(index, self.catalog.sounds.len() - 1)? as u32)
        }

        fn decode_draw(&mut self, lane: usize, draw: &mut Draw) -> Result<()> {
            let _ = &self.packet[lane..lane + DRAW_FLOATS];
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
            let _ = &self.packet[lane..lane + SOLID_FACE_FLOATS];
            let normal = Vector { x: self.packet[lane], y: self.packet[lane + 1], z: self.packet[lane + 2] };
            let distance = self.packet[lane + 3];
            let material = self.material_at(lane + 4)?;
            *solid_face = SolidFace { normal, distance, material };
            Ok(())
        }

        fn decode_decal(&mut self, lane: usize, decal: &mut Decal) -> Result<()> {
            let _ = &self.packet[lane..lane + DECAL_FLOATS];
            let material = self.material_at(lane)?;
            let transform = self.matrix(lane + 1);
            let opacity = self.packet[lane + 17];
            *decal = Decal { material, transform, opacity };
            Ok(())
        }

        fn decode_sound(&mut self, lane: usize, sound: &mut Sound) -> Result<()> {
            let _ = &self.packet[lane..lane + SOUND_FLOATS];
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
            let _ = &self.packet[lane..lane + SPRITE_FLOATS];
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
            let _ = &self.packet[lane..lane + INSET_FLOATS];
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
            let after_sprites = self.count(lane + 47, MAX_SPRITES)?;
            *inset = Inset { x, y, width, height, view, projection, ambient, light_direction, light_color, draws, pixel_size, after_sprites: after_sprites as u32 };
            Ok(())
        }

        fn decode_bus(&mut self, lane: usize, bus: &mut Bus) -> Result<()> {
            let _ = &self.packet[lane..lane + BUS_FLOATS];
            let gain = self.packet[lane];
            let lowpass = self.packet[lane + 1];
            let reverb = self.packet[lane + 2];
            let comp_threshold = self.packet[lane + 3];
            let comp_ratio = self.packet[lane + 4];
            let duck_bus = self.count(lane + 5, 15)?;
            let duck_amount = self.packet[lane + 6];
            let eq_low_db = self.packet[lane + 7];
            let eq_mid_db = self.packet[lane + 8];
            let eq_high_db = self.packet[lane + 9];
            *bus = Bus { gain, lowpass, reverb, comp_threshold, comp_ratio, duck_bus: duck_bus as u32, duck_amount, eq_low_db, eq_mid_db, eq_high_db };
            Ok(())
        }

        fn decode_voice(&mut self, lane: usize, voice: &mut Voice) -> Result<()> {
            let _ = &self.packet[lane..lane + VOICE_FLOATS];
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

        fn decode_pose(&mut self, lane: usize, pose: &mut Pose) -> Result<()> {
            let _ = &self.packet[lane..lane + POSE_FLOATS];
            let draw = self.count(lane, MAX_DRAWS)?;
            let clip = self.count(lane + 1, 65535)?;
            let time = self.packet[lane + 2];
            let weight = self.packet[lane + 3];
            let inset = self.count(lane + 4, 16)?;
            *pose = Pose { draw: draw as u32, clip: clip as u32, time, weight, inset: inset as u32 };
            Ok(())
        }

        fn decode_emitter(&mut self, lane: usize, emitter: &mut Emitter) -> Result<()> {
            let _ = &self.packet[lane..lane + EMITTER_FLOATS];
            let material = self.material_at(lane)?;
            let transform = self.matrix(lane + 1);
            let seed = self.count(lane + 17, 16777215)?;
            let time = self.packet[lane + 18];
            let duration = self.packet[lane + 19];
            let rate = self.packet[lane + 20];
            let lifetime = self.packet[lane + 21];
            let radius = self.packet[lane + 22];
            let velocity = Vector { x: self.packet[lane + 23], y: self.packet[lane + 24], z: self.packet[lane + 25] };
            let spread = self.packet[lane + 26];
            let gravity = Vector { x: self.packet[lane + 27], y: self.packet[lane + 28], z: self.packet[lane + 29] };
            let drag = self.packet[lane + 30];
            let size_start = self.packet[lane + 31];
            let size_end = self.packet[lane + 32];
            let color_start = Color { r: self.packet[lane + 33], g: self.packet[lane + 34], b: self.packet[lane + 35], a: self.packet[lane + 36] };
            let color_end = Color { r: self.packet[lane + 37], g: self.packet[lane + 38], b: self.packet[lane + 39], a: self.packet[lane + 40] };
            let additive = self.flag(lane + 41)?;
            let soft_depth = self.packet[lane + 42];
            let lit = self.packet[lane + 43];
            let flipbook_cols = self.count(lane + 44, 32)?;
            let flipbook_rows = self.count(lane + 45, 32)?;
            let flipbook_fps = self.packet[lane + 46];
            *emitter = Emitter { material, transform, seed: seed as u32, time, duration, rate, lifetime, radius, velocity, spread, gravity, drag, size_start, size_end, color_start, color_end, additive, soft_depth, lit, flipbook_cols: flipbook_cols as u32, flipbook_rows: flipbook_rows as u32, flipbook_fps };
            Ok(())
        }

        fn decode_light(&mut self, lane: usize, light: &mut Light) -> Result<()> {
            let _ = &self.packet[lane..lane + LIGHT_FLOATS];
            let position = Vector { x: self.packet[lane], y: self.packet[lane + 1], z: self.packet[lane + 2] };
            let radius = self.packet[lane + 3];
            let color = Vector { x: self.packet[lane + 4], y: self.packet[lane + 5], z: self.packet[lane + 6] };
            let direction = Vector { x: self.packet[lane + 7], y: self.packet[lane + 8], z: self.packet[lane + 9] };
            let inner = self.packet[lane + 10];
            let outer = self.packet[lane + 11];
            let shadow = self.flag(lane + 12)?;
            let id = self.count(lane + 13, 16777215)?;
            *light = Light { position, radius, color, direction, inner, outer, shadow, id: id as u32 };
            Ok(())
        }

        fn decode_cascade(&mut self, lane: usize, cascade: &mut Cascade) -> Result<()> {
            let _ = &self.packet[lane..lane + CASCADE_FLOATS];
            let projection = self.matrix(lane);
            let texel = self.packet[lane + 16];
            let far = self.packet[lane + 17];
            *cascade = Cascade { projection, texel, far };
            Ok(())
        }

        fn decode_probe(&mut self, lane: usize, probe: &mut Probe) -> Result<()> {
            let _ = &self.packet[lane..lane + PROBE_FLOATS];
            let sh0 = Vector { x: self.packet[lane], y: self.packet[lane + 1], z: self.packet[lane + 2] };
            let sh1 = Vector { x: self.packet[lane + 3], y: self.packet[lane + 4], z: self.packet[lane + 5] };
            let sh2 = Vector { x: self.packet[lane + 6], y: self.packet[lane + 7], z: self.packet[lane + 8] };
            let sh3 = Vector { x: self.packet[lane + 9], y: self.packet[lane + 10], z: self.packet[lane + 11] };
            let sh4 = Vector { x: self.packet[lane + 12], y: self.packet[lane + 13], z: self.packet[lane + 14] };
            let sh5 = Vector { x: self.packet[lane + 15], y: self.packet[lane + 16], z: self.packet[lane + 17] };
            let sh6 = Vector { x: self.packet[lane + 18], y: self.packet[lane + 19], z: self.packet[lane + 20] };
            let sh7 = Vector { x: self.packet[lane + 21], y: self.packet[lane + 22], z: self.packet[lane + 23] };
            let sh8 = Vector { x: self.packet[lane + 24], y: self.packet[lane + 25], z: self.packet[lane + 26] };
            let valid = self.packet[lane + 27];
            *probe = Probe { sh0, sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8, valid };
            Ok(())
        }

        fn decode_reflection_probe(&mut self, lane: usize, reflection_probe: &mut ReflectionProbe) -> Result<()> {
            let _ = &self.packet[lane..lane + REFLECTION_PROBE_FLOATS];
            let id = self.count(lane, 16777215)?;
            let revision = self.count(lane + 1, 16777215)?;
            let priority = self.count(lane + 2, 255)?;
            let capture = self.flag(lane + 3)?;
            let position = Vector { x: self.packet[lane + 4], y: self.packet[lane + 5], z: self.packet[lane + 6] };
            let center = Vector { x: self.packet[lane + 7], y: self.packet[lane + 8], z: self.packet[lane + 9] };
            let extent = Vector { x: self.packet[lane + 10], y: self.packet[lane + 11], z: self.packet[lane + 12] };
            let blend = self.packet[lane + 13];
            let parallax_center = Vector { x: self.packet[lane + 14], y: self.packet[lane + 15], z: self.packet[lane + 16] };
            let parallax_extent = Vector { x: self.packet[lane + 17], y: self.packet[lane + 18], z: self.packet[lane + 19] };
            let far = self.packet[lane + 20];
            *reflection_probe = ReflectionProbe { id: id as u32, revision: revision as u32, priority: priority as u32, capture, position, center, extent, blend, parallax_center, parallax_extent, far };
            Ok(())
        }

        fn decode_fog_volume(&mut self, lane: usize, fog_volume: &mut FogVolume) -> Result<()> {
            let _ = &self.packet[lane..lane + FOG_VOLUME_FLOATS];
            let shape = self.count(lane, 1)?;
            let center = Vector { x: self.packet[lane + 1], y: self.packet[lane + 2], z: self.packet[lane + 3] };
            let extent = Vector { x: self.packet[lane + 4], y: self.packet[lane + 5], z: self.packet[lane + 6] };
            let density = self.packet[lane + 7];
            let albedo = Vector { x: self.packet[lane + 8], y: self.packet[lane + 9], z: self.packet[lane + 10] };
            let emission = Vector { x: self.packet[lane + 11], y: self.packet[lane + 12], z: self.packet[lane + 13] };
            let falloff = self.packet[lane + 14];
            *fog_volume = FogVolume { shape: shape as u32, center, extent, density, albedo, emission, falloff };
            Ok(())
        }

        fn decode_listener(&mut self, lane: usize, listener: &mut Listener) -> Result<()> {
            let _ = &self.packet[lane..lane + LISTENER_FLOATS];
            let position = Vector { x: self.packet[lane], y: self.packet[lane + 1], z: self.packet[lane + 2] };
            let forward = Vector { x: self.packet[lane + 3], y: self.packet[lane + 4], z: self.packet[lane + 5] };
            let up = Vector { x: self.packet[lane + 6], y: self.packet[lane + 7], z: self.packet[lane + 8] };
            let velocity = Vector { x: self.packet[lane + 9], y: self.packet[lane + 10], z: self.packet[lane + 11] };
            *listener = Listener { position, forward, up, velocity };
            Ok(())
        }

        fn decode_panner(&mut self, lane: usize, panner: &mut Panner) -> Result<()> {
            let _ = &self.packet[lane..lane + PANNER_FLOATS];
            let voice = self.count(lane, MAX_VOICE_ID)?;
            let sound = self.count(lane + 1, 255)?;
            let position = Vector { x: self.packet[lane + 2], y: self.packet[lane + 3], z: self.packet[lane + 4] };
            let orientation = Vector { x: self.packet[lane + 5], y: self.packet[lane + 6], z: self.packet[lane + 7] };
            let distance_model = self.count(lane + 8, 2)?;
            let ref_distance = self.packet[lane + 9];
            let max_distance = self.packet[lane + 10];
            let rolloff = self.packet[lane + 11];
            let cone_inner = self.packet[lane + 12];
            let cone_outer = self.packet[lane + 13];
            let cone_outer_gain = self.packet[lane + 14];
            let hrtf = self.flag(lane + 15)?;
            let velocity = Vector { x: self.packet[lane + 16], y: self.packet[lane + 17], z: self.packet[lane + 18] };
            let doppler = self.packet[lane + 19];
            let occlusion = self.packet[lane + 20];
            let zone = self.count(lane + 21, 64)?;
            *panner = Panner { voice: voice as u32, sound: sound as u32, position, orientation, distance_model: distance_model as u32, ref_distance, max_distance, rolloff, cone_inner, cone_outer, cone_outer_gain, hrtf, velocity, doppler, occlusion, zone: zone as u32 };
            Ok(())
        }

        fn decode_instance(&mut self, lane: usize, instance: &mut Instance) -> Result<()> {
            let _ = &self.packet[lane..lane + INSTANCE_FLOATS];
            let id_low = self.count(lane, 65535)?;
            let id_high = self.count(lane + 1, 65535)?;
            let order = self.count(lane + 2, 16777215)?;
            let mesh = self.mesh_at(lane + 3)?;
            let material = self.material_at(lane + 4)?;
            let transform = self.matrix(lane + 5);
            let parameters_count = self.count(lane + 21, 4)?;
            ensure!(parameters_count == 0 || parameters_count == 4, "draw parameters must contain zero or four lanes");
            let parameters = (parameters_count == 4).then(|| self.packet[lane + 22..lane + 26].try_into().unwrap());
            *instance = Instance { id_low: id_low as u32, id_high: id_high as u32, order: order as u32, mesh, material, transform, parameters };
            Ok(())
        }

        fn decode_instance_removal(&mut self, lane: usize, instance_removal: &mut InstanceRemoval) -> Result<()> {
            let _ = &self.packet[lane..lane + INSTANCE_REMOVAL_FLOATS];
            let id_low = self.count(lane, 65535)?;
            let id_high = self.count(lane + 1, 65535)?;
            *instance_removal = InstanceRemoval { id_low: id_low as u32, id_high: id_high as u32 };
            Ok(())
        }

        fn decode_bone_matrix(&mut self, lane: usize, bone_matrix: &mut BoneMatrix) -> Result<()> {
            let _ = &self.packet[lane..lane + BONE_MATRIX_FLOATS];
            let inset = self.count(lane, 16)?;
            let draw = self.count(lane + 1, MAX_DRAWS)?;
            let joint = self.count(lane + 2, 1024)?;
            let matrix = self.matrix(lane + 3);
            *bone_matrix = BoneMatrix { inset: inset as u32, draw: draw as u32, joint: joint as u32, matrix };
            Ok(())
        }

        fn decode_morph_weight(&mut self, lane: usize, morph_weight: &mut MorphWeight) -> Result<()> {
            let _ = &self.packet[lane..lane + MORPH_WEIGHT_FLOATS];
            let inset = self.count(lane, 16)?;
            let draw = self.count(lane + 1, MAX_DRAWS)?;
            let target = self.count(lane + 2, 256)?;
            let weight = self.packet[lane + 3];
            *morph_weight = MorphWeight { inset: inset as u32, draw: draw as u32, target: target as u32, weight };
            Ok(())
        }

        fn decode_motion(&mut self, lane: usize, motion: &mut Motion) -> Result<()> {
            let _ = &self.packet[lane..lane + MOTION_FLOATS];
            let draw = self.count(lane, MAX_DRAWS)?;
            let previous = self.matrix(lane + 1);
            *motion = Motion { draw: draw as u32, previous };
            Ok(())
        }

        fn decode_ribbon_point(&mut self, lane: usize, ribbon_point: &mut RibbonPoint) -> Result<()> {
            let _ = &self.packet[lane..lane + RIBBON_FLOATS];
            let ribbon = self.count(lane, 65535)?;
            let material = self.material_at(lane + 1)?;
            let additive = self.flag(lane + 2)?;
            let position = Vector { x: self.packet[lane + 3], y: self.packet[lane + 4], z: self.packet[lane + 5] };
            let width = self.packet[lane + 6];
            let color = Color { r: self.packet[lane + 7], g: self.packet[lane + 8], b: self.packet[lane + 9], a: self.packet[lane + 10] };
            let soft_depth = self.packet[lane + 11];
            *ribbon_point = RibbonPoint { ribbon: ribbon as u32, material, additive, position, width, color, soft_depth };
            Ok(())
        }

        fn decode_haptic(&mut self, lane: usize, haptic: &mut Haptic) -> Result<()> {
            let _ = &self.packet[lane..lane + HAPTIC_FLOATS];
            let pad = self.count(lane, 15)?;
            let low = self.packet[lane + 1];
            let high = self.packet[lane + 2];
            let duration = self.packet[lane + 3];
            *haptic = Haptic { pad: pad as u32, low, high, duration };
            Ok(())
        }

        fn decode_reverb_zone(&mut self, lane: usize, reverb_zone: &mut ReverbZone) -> Result<()> {
            let _ = &self.packet[lane..lane + REVERB_ZONE_FLOATS];
            let id = self.count(lane, 64)?;
            let position = Vector { x: self.packet[lane + 1], y: self.packet[lane + 2], z: self.packet[lane + 3] };
            let half_extents = Vector { x: self.packet[lane + 4], y: self.packet[lane + 5], z: self.packet[lane + 6] };
            let blend = self.packet[lane + 7];
            let decay = self.packet[lane + 8];
            let damping = self.packet[lane + 9];
            let wet = self.packet[lane + 10];
            let portal_attenuation = self.packet[lane + 11];
            *reverb_zone = ReverbZone { id: id as u32, position, half_extents, blend, decay, damping, wet, portal_attenuation };
            Ok(())
        }

        fn decode_synth_voice(&mut self, lane: usize, synth_voice: &mut SynthVoice) -> Result<()> {
            let _ = &self.packet[lane..lane + SYNTH_VOICE_FLOATS];
            let id = self.count(lane, MAX_VOICE_ID)?;
            let mode = self.count(lane + 1, 5)?;
            let bus = self.count(lane + 2, 15)?;
            let frequency = self.packet[lane + 3];
            let gain = self.packet[lane + 4];
            let pan = self.packet[lane + 5];
            let detune = self.packet[lane + 6];
            let attack = self.packet[lane + 7];
            let decay = self.packet[lane + 8];
            let sustain = self.packet[lane + 9];
            let release = self.packet[lane + 10];
            let cutoff = self.packet[lane + 11];
            let resonance = self.packet[lane + 12];
            let grain_size = self.packet[lane + 13];
            let grain_pitch = self.packet[lane + 14];
            let reverb = self.packet[lane + 15];
            *synth_voice = SynthVoice { id: id as u32, mode: mode as u32, bus: bus as u32, frequency, gain, pan, detune, attack, decay, sustain, release, cutoff, resonance, grain_size, grain_pitch, reverb };
            Ok(())
        }

        fn decode_area_light_spec(&mut self, lane: usize, area_light_spec: &mut AreaLightSpec) -> Result<()> {
            let _ = &self.packet[lane..lane + AREA_LIGHT_FLOATS];
            let kind = self.count(lane, 3)?;
            let position = Vector { x: self.packet[lane + 1], y: self.packet[lane + 2], z: self.packet[lane + 3] };
            let right = Vector { x: self.packet[lane + 4], y: self.packet[lane + 5], z: self.packet[lane + 6] };
            let up = Vector { x: self.packet[lane + 7], y: self.packet[lane + 8], z: self.packet[lane + 9] };
            let color = Vector { x: self.packet[lane + 10], y: self.packet[lane + 11], z: self.packet[lane + 12] };
            let intensity = self.packet[lane + 13];
            let width = self.packet[lane + 14];
            let height = self.packet[lane + 15];
            let range = self.packet[lane + 16];
            let cookie_material = self.count(lane + 17, 65535)?;
            let two_sided = self.flag(lane + 18)?;
            *area_light_spec = AreaLightSpec { kind: kind as u32, position, right, up, color, intensity, width, height, range, cookie_material: cookie_material as u32, two_sided };
            Ok(())
        }

        fn decode_pbr_decal_spec(&mut self, lane: usize, pbr_decal_spec: &mut PbrDecalSpec) -> Result<()> {
            let _ = &self.packet[lane..lane + PBR_DECAL_FLOATS];
            let material = self.material_at(lane)?;
            let transform = self.matrix(lane + 1);
            let albedo = Color { r: self.packet[lane + 17], g: self.packet[lane + 18], b: self.packet[lane + 19], a: self.packet[lane + 20] };
            let normal_strength = self.packet[lane + 21];
            let roughness = self.packet[lane + 22];
            let metallic = self.packet[lane + 23];
            let emissive_strength = self.packet[lane + 24];
            let fade = self.packet[lane + 25];
            *pbr_decal_spec = PbrDecalSpec { material, transform, albedo, normal_strength, roughness, metallic, emissive_strength, fade };
            Ok(())
        }

        fn decode_camera_target_spec(&mut self, lane: usize, camera_target_spec: &mut CameraTargetSpec) -> Result<()> {
            let _ = &self.packet[lane..lane + CAMERA_TARGET_FLOATS];
            let material = self.material_at(lane)?;
            let width = self.count(lane + 1, 4096)?;
            let height = self.count(lane + 2, 4096)?;
            let viewport_x = self.packet[lane + 3];
            let viewport_y = self.packet[lane + 4];
            let viewport_w = self.packet[lane + 5];
            let viewport_h = self.packet[lane + 6];
            let split_screen = self.flag(lane + 7)?;
            let view = self.matrix(lane + 8);
            let projection = self.matrix(lane + 24);
            *camera_target_spec = CameraTargetSpec { material, width: width as u32, height: height as u32, viewport_x, viewport_y, viewport_w, viewport_h, split_screen, view, projection };
            Ok(())
        }

        fn decode_video_stream_spec(&mut self, lane: usize, video_stream_spec: &mut VideoStreamSpec) -> Result<()> {
            let _ = &self.packet[lane..lane + VIDEO_STREAM_FLOATS];
            let material = self.material_at(lane)?;
            let time = self.packet[lane + 1];
            let playing = self.flag(lane + 2)?;
            let looping = self.flag(lane + 3)?;
            let speed = self.packet[lane + 4];
            *video_stream_spec = VideoStreamSpec { material, time, playing, looping, speed };
            Ok(())
        }
    }

    /// Decodes a render packet. The game's frame: what to draw and play, and what to do.
    /// Failures include the earliest trustworthy structural context.
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
        decode_raw(packet, schema, catalog, frame).map_err(|error| error.context(explain_frame(packet, schema, catalog)))
    }

    pub(crate) fn decode_raw<'a>(packet: &'a [f32], schema: u32, catalog: &Catalog, frame: &mut RenderFrame) -> Result<&'a [f32]> {
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
        let fields_count = reader.count(157, MAX_FIELD_PATCHES)?;
        let fields_lanes = reader.count(158, 16805888)?;
        let ui_count = reader.count(159, MAX_UI_NODES)?;
        let ui_lanes = reader.count(160, 4292608)?;
        let services_count = reader.count(161, MAX_SERVICE_REQUESTS)?;
        let services_lanes = reader.count(162, 4196352)?;
        let buses_count = reader.count(163, MAX_BUSES)?;
        let voices_count = reader.count(164, MAX_VOICES)?;
        let poses_count = reader.count(165, MAX_POSES)?;
        let emitters_count = reader.count(166, MAX_EMITTERS)?;
        let lights_count = reader.count(168, MAX_LIGHTS)?;
        let cascades_count = reader.count(169, MAX_CASCADES)?;
        let probes_count = reader.count(170, MAX_PROBES)?;
        let reflection_probes_count = reader.count(171, MAX_REFLECTION_PROBES)?;
        let listener_count = reader.count(172, 1)?;
        let panners_count = reader.count(173, MAX_PANNERS)?;
        let instances_count = reader.count(179, MAX_INSTANCES)?;
        let instance_removals_count = reader.count(180, MAX_INSTANCE_REMOVALS)?;
        let fog_volumes_count = reader.count(234, MAX_FOG_VOLUMES)?;
        let bone_matrices_count = reader.count(235, MAX_BONE_MATRICES)?;
        let morph_weights_count = reader.count(236, MAX_MORPH_WEIGHTS)?;
        let motions_count = reader.count(244, MAX_MOTIONS)?;
        let ribbons_count = reader.count(246, MAX_RIBBONS)?;
        let haptics_count = reader.count(249, MAX_HAPTICS)?;
        let reverb_zones_count = reader.count(250, MAX_REVERB_ZONES)?;
        let synths_count = reader.count(251, MAX_SYNTHS)?;
        let area_lights_count = reader.count(262, MAX_AREA_LIGHTS)?;
        let pbr_decals_count = reader.count(263, MAX_PBR_DECALS)?;
        let camera_targets_count = reader.count(264, MAX_CAMERA_TARGETS)?;
        let video_streams_count = reader.count(265, MAX_VIDEO_STREAMS)?;
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
        let poses_start = voices_end;
        let poses_end = poses_start + poses_count * POSE_FLOATS;
        let emitters_start = poses_end;
        let emitters_end = emitters_start + emitters_count * EMITTER_FLOATS;
        let lights_start = emitters_end;
        let lights_end = lights_start + lights_count * LIGHT_FLOATS;
        let cascades_start = lights_end;
        let cascades_end = cascades_start + cascades_count * CASCADE_FLOATS;
        let probes_start = cascades_end;
        let probes_end = probes_start + probes_count * PROBE_FLOATS;
        let reflection_probes_start = probes_end;
        let reflection_probes_end = reflection_probes_start + reflection_probes_count * REFLECTION_PROBE_FLOATS;
        let fog_volumes_start = reflection_probes_end;
        let fog_volumes_end = fog_volumes_start + fog_volumes_count * FOG_VOLUME_FLOATS;
        let listener_start = fog_volumes_end;
        let listener_end = listener_start + listener_count * LISTENER_FLOATS;
        let panners_start = listener_end;
        let panners_end = panners_start + panners_count * PANNER_FLOATS;
        let instances_start = panners_end;
        let instances_end = instances_start + instances_count * INSTANCE_FLOATS;
        let instance_removals_start = instances_end;
        let instance_removals_end = instance_removals_start + instance_removals_count * INSTANCE_REMOVAL_FLOATS;
        let bone_matrices_start = instance_removals_end;
        let bone_matrices_end = bone_matrices_start + bone_matrices_count * BONE_MATRIX_FLOATS;
        let morph_weights_start = bone_matrices_end;
        let morph_weights_end = morph_weights_start + morph_weights_count * MORPH_WEIGHT_FLOATS;
        let motions_start = morph_weights_end;
        let motions_end = motions_start + motions_count * MOTION_FLOATS;
        let ribbons_start = motions_end;
        let ribbons_end = ribbons_start + ribbons_count * RIBBON_FLOATS;
        let haptics_start = ribbons_end;
        let haptics_end = haptics_start + haptics_count * HAPTIC_FLOATS;
        let reverb_zones_start = haptics_end;
        let reverb_zones_end = reverb_zones_start + reverb_zones_count * REVERB_ZONE_FLOATS;
        let synths_start = reverb_zones_end;
        let synths_end = synths_start + synths_count * SYNTH_VOICE_FLOATS;
        let area_lights_start = synths_end;
        let area_lights_end = area_lights_start + area_lights_count * AREA_LIGHT_FLOATS;
        let pbr_decals_start = area_lights_end;
        let pbr_decals_end = pbr_decals_start + pbr_decals_count * PBR_DECAL_FLOATS;
        let camera_targets_start = pbr_decals_end;
        let camera_targets_end = camera_targets_start + camera_targets_count * CAMERA_TARGET_FLOATS;
        let video_streams_start = camera_targets_end;
        let video_streams_end = video_streams_start + video_streams_count * VIDEO_STREAM_FLOATS;
        let extras_start = video_streams_end;
        let truncated = "game returned a truncated, oversized or nonfinite frame";
        ensure!(packet.len() >= video_streams_end && all_finite(packet), truncated);
        // What the records' tails claim of extras, in order.
        let mut extras_lanes = 0;
        for index in 0..draws_count {
            extras_lanes += reader.count(draws_start + index * DRAW_FLOATS + 23, MAX_DRAW_EXTRA)? * 4;
        }
        for index in 0..inset_draws_count {
            extras_lanes += reader.count(inset_draws_start + index * DRAW_FLOATS + 23, MAX_DRAW_EXTRA)? * 4;
        }
        let extras_end = extras_start + extras_lanes;
        // The game's own sections, after the rest: the marker, their count, then each
        // one's tag, length and values.
        let mut extensions = std::mem::take(&mut frame.extensions);
        let mut ext_len = 0usize;
        let mut extensions_end = extras_end;
        #[allow(unused_mut)]
        let mut provenance_seen = false;
        if packet.len() > extensions_end {
            // Lanes past the end that are no block are an oversized packet.
            ensure!(packet.len() >= extensions_end + 2 && packet[extensions_end] == EXTENSIONS_MARKER, truncated);
            let invalid = "game returned invalid extension sections";
            let count = reader.count(extensions_end + 1, MAX_EXTENSIONS)?;
            extensions_end += 2;
            for _ in 0..count {
                ensure!(packet.len() >= extensions_end + 2, invalid);
                let tag = reader.count(extensions_end, MAX_EXTENSION_TAG)?;
                let length = reader.count(extensions_end + 1, MAX_EXTENSION_LANES)?;
                let first = extensions_end + 2;
                ensure!(packet.len() >= first + length, invalid);
                if tag == PROVENANCE_TAG {
                    #[cfg(feature = "diagnostics")]
                    {
                        ensure!(!provenance_seen, "duplicate development provenance");
                        provenance_seen = true;
                        crate::diagnostic::read_provenance(&packet[first..first + length], extras_end).map_err(|_| anyhow::anyhow!("invalid development provenance"))?;
                    }
                    #[cfg(not(feature = "diagnostics"))]
                    {
                        let _ = provenance_seen;
                        anyhow::bail!("development provenance requires a host built with diagnostics");
                    }
                } else {
                    if let Some(slot) = extensions.get_mut(ext_len) {
                        slot.tag = tag as u32;
                        slot.values.clear();
                        slot.values.extend_from_slice(&packet[first..first + length]);
                    } else {
                        extensions.push(Extension { tag: tag as u32, values: packet[first..first + length].to_vec() });
                    }
                    ext_len += 1;
                }
                extensions_end = first + length;
            }
        }
        extensions.truncate(ext_len);
        ensure!(packet.len() == extensions_end, truncated);
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
        let sky_tint = Vector { x: reader.packet[87], y: reader.packet[88], z: reader.packet[89] };
        let sky_sh0 = Vector { x: reader.packet[90], y: reader.packet[91], z: reader.packet[92] };
        let sky_sh1 = Vector { x: reader.packet[93], y: reader.packet[94], z: reader.packet[95] };
        let sky_sh2 = Vector { x: reader.packet[96], y: reader.packet[97], z: reader.packet[98] };
        let sky_sh3 = Vector { x: reader.packet[99], y: reader.packet[100], z: reader.packet[101] };
        let sky_sh4 = Vector { x: reader.packet[102], y: reader.packet[103], z: reader.packet[104] };
        let sky_sh5 = Vector { x: reader.packet[105], y: reader.packet[106], z: reader.packet[107] };
        let sky_sh6 = Vector { x: reader.packet[108], y: reader.packet[109], z: reader.packet[110] };
        let sky_sh7 = Vector { x: reader.packet[111], y: reader.packet[112], z: reader.packet[113] };
        let sky_sh8 = Vector { x: reader.packet[114], y: reader.packet[115], z: reader.packet[116] };
        let probe_volume_revision = reader.count(117, 16777215)?;
        let probe_volume_nx = reader.count(118, 64)?;
        let probe_volume_ny = reader.count(119, 64)?;
        let probe_volume_nz = reader.count(120, 64)?;
        let probe_volume_spacing = reader.packet[121];
        let probe_volume_origin = Vector { x: reader.packet[122], y: reader.packet[123], z: reader.packet[124] };
        let fog_radius = reader.packet[125];
        let fog_density = reader.packet[126];
        let fog_height = reader.packet[127];
        let shadow_view = reader.matrix(128);
        let shadow_texels = reader.count(144, 8192)?;
        let shadow_filter = reader.count(145, 2)?;
        let shadow_softness = reader.packet[146];
        let shadow_contact = reader.count(147, 64)?;
        let shadow_contact_length = reader.packet[148];
        let shadow_local = reader.count(149, 16)?;
        let shadow_local_texels = reader.count(150, 4096)?;
        let blur = reader.packet[151];
        let quality_bloom = reader.flag(152)?;
        let clouds_cover = reader.packet[153];
        let clouds_drift = Drift { x: reader.packet[154], z: reader.packet[155] };
        let hide_cursor = reader.flag(156)?;
        let quality_taa = reader.flag(167)?;
        let loading = reader.flag(174)?;
        let show_captions = reader.flag(175)?;
        let retained_enabled = reader.flag(176)?;
        let retained_reset = reader.flag(177)?;
        let retained_after = reader.count(178, 100000)?;
        let exposure_auto = reader.flag(181)?;
        let exposure_compensation = reader.packet[182];
        let exposure_min = reader.packet[183];
        let exposure_max = reader.packet[184];
        let exposure_brighten = reader.packet[185];
        let exposure_darken = reader.packet[186];
        let exposure_low = reader.packet[187];
        let exposure_high = reader.packet[188];
        let exposure_centre = reader.packet[189];
        let exposure_range_low = reader.packet[190];
        let exposure_range_high = reader.packet[191];
        let exposure_target = reader.packet[192];
        let exposure_dt = reader.packet[193];
        let exposure_cut = reader.flag(194)?;
        let grade_temperature = reader.packet[195];
        let grade_tint = reader.packet[196];
        let grade_exposure = reader.packet[197];
        let grade_contrast = reader.packet[198];
        let grade_pivot = reader.packet[199];
        let grade_saturation = reader.packet[200];
        let grade_vibrance = reader.packet[201];
        let grade_lift = Vector { x: reader.packet[202], y: reader.packet[203], z: reader.packet[204] };
        let grade_gamma = Vector { x: reader.packet[205], y: reader.packet[206], z: reader.packet[207] };
        let grade_gain = Vector { x: reader.packet[208], y: reader.packet[209], z: reader.packet[210] };
        let grade_lut_a = reader.count(211, 16777215)?;
        let grade_lut_b = reader.count(212, 16777215)?;
        let grade_lut_blend = reader.packet[213];
        let grade_lut_weight = reader.packet[214];
        let grade_vignette = reader.packet[215];
        let grade_vignette_radius = reader.packet[216];
        let quality_ssr = reader.count(217, 2)?;
        let reflection_budget = reader.count(218, 6)?;
        let fog_edge = reader.packet[219];
        let fog_distance = reader.packet[220];
        let fog_albedo = Vector { x: reader.packet[221], y: reader.packet[222], z: reader.packet[223] };
        let fog_anisotropy = reader.packet[224];
        let fog_sun = reader.packet[225];
        let fog_ambient = reader.packet[226];
        let fog_sky = reader.packet[227];
        let fog_noise = reader.packet[228];
        let fog_noise_scale = reader.packet[229];
        let fog_drift = Vector { x: reader.packet[230], y: reader.packet[231], z: reader.packet[232] };
        let quality_volumetrics = reader.count(233, 2)?;
        let motion_blur_strength = reader.packet[237];
        let motion_blur_max_radius = reader.packet[238];
        let motion_blur_samples = reader.count(239, 32)?;
        let dof_focus_distance = reader.packet[240];
        let dof_focus_range = reader.packet[241];
        let dof_aperture = reader.packet[242];
        let dof_max_blur = reader.packet[243];
        let quality_ssgi = reader.flag(245)?;
        let reverb_decay = reader.packet[247];
        let reverb_damping = reader.packet[248];
        let grade_tonemap = reader.count(252, 3)?;
        let grade_sharpness = reader.packet[253];
        let grade_grain = reader.packet[254];
        let grade_flare = reader.packet[255];
        let grade_outline_strength = reader.packet[256];
        let quality_oit = reader.flag(257)?;
        let quality_physical_sky = reader.flag(258)?;
        let quality_upscale = reader.flag(259)?;
        let quality_ray_tracing = reader.count(260, 3)?;
        let quality_ddgi = reader.flag(261)?;
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
            let service = reader.count(service_lane + 1, 15)?;
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
        let mut poses = std::mem::take(&mut frame.poses);
        resize(&mut poses, poses_count);
        for (index, pose) in poses.iter_mut().enumerate() {
            reader.decode_pose(poses_start + index * POSE_FLOATS, pose)?;
        }
        let mut emitters = std::mem::take(&mut frame.emitters);
        resize(&mut emitters, emitters_count);
        for (index, emitter) in emitters.iter_mut().enumerate() {
            reader.decode_emitter(emitters_start + index * EMITTER_FLOATS, emitter)?;
        }
        let mut lights = std::mem::take(&mut frame.lights);
        resize(&mut lights, lights_count);
        for (index, light) in lights.iter_mut().enumerate() {
            reader.decode_light(lights_start + index * LIGHT_FLOATS, light)?;
        }
        let mut cascades = std::mem::take(&mut frame.cascades);
        resize(&mut cascades, cascades_count);
        for (index, cascade) in cascades.iter_mut().enumerate() {
            reader.decode_cascade(cascades_start + index * CASCADE_FLOATS, cascade)?;
        }
        let mut probes = std::mem::take(&mut frame.probes);
        resize(&mut probes, probes_count);
        for (index, probe) in probes.iter_mut().enumerate() {
            reader.decode_probe(probes_start + index * PROBE_FLOATS, probe)?;
        }
        let mut reflection_probes = std::mem::take(&mut frame.reflection_probes);
        resize(&mut reflection_probes, reflection_probes_count);
        for (index, reflection_probe) in reflection_probes.iter_mut().enumerate() {
            reader.decode_reflection_probe(reflection_probes_start + index * REFLECTION_PROBE_FLOATS, reflection_probe)?;
        }
        let mut fog_volumes = std::mem::take(&mut frame.fog_volumes);
        resize(&mut fog_volumes, fog_volumes_count);
        for (index, fog_volume) in fog_volumes.iter_mut().enumerate() {
            reader.decode_fog_volume(fog_volumes_start + index * FOG_VOLUME_FLOATS, fog_volume)?;
        }
        let mut listener = std::mem::take(&mut frame.listener);
        resize(&mut listener, listener_count);
        for (index, listener) in listener.iter_mut().enumerate() {
            reader.decode_listener(listener_start + index * LISTENER_FLOATS, listener)?;
        }
        let mut panners = std::mem::take(&mut frame.panners);
        resize(&mut panners, panners_count);
        for (index, panner) in panners.iter_mut().enumerate() {
            reader.decode_panner(panners_start + index * PANNER_FLOATS, panner)?;
        }
        let mut instances = std::mem::take(&mut frame.instances);
        resize(&mut instances, instances_count);
        for (index, instance) in instances.iter_mut().enumerate() {
            reader.decode_instance(instances_start + index * INSTANCE_FLOATS, instance)?;
        }
        let mut instance_removals = std::mem::take(&mut frame.instance_removals);
        resize(&mut instance_removals, instance_removals_count);
        for (index, instance_removal) in instance_removals.iter_mut().enumerate() {
            reader.decode_instance_removal(instance_removals_start + index * INSTANCE_REMOVAL_FLOATS, instance_removal)?;
        }
        let mut bone_matrices = std::mem::take(&mut frame.bone_matrices);
        resize(&mut bone_matrices, bone_matrices_count);
        for (index, bone_matrix) in bone_matrices.iter_mut().enumerate() {
            reader.decode_bone_matrix(bone_matrices_start + index * BONE_MATRIX_FLOATS, bone_matrix)?;
        }
        let mut morph_weights = std::mem::take(&mut frame.morph_weights);
        resize(&mut morph_weights, morph_weights_count);
        for (index, morph_weight) in morph_weights.iter_mut().enumerate() {
            reader.decode_morph_weight(morph_weights_start + index * MORPH_WEIGHT_FLOATS, morph_weight)?;
        }
        let mut motions = std::mem::take(&mut frame.motions);
        resize(&mut motions, motions_count);
        for (index, motion) in motions.iter_mut().enumerate() {
            reader.decode_motion(motions_start + index * MOTION_FLOATS, motion)?;
        }
        let mut ribbons = std::mem::take(&mut frame.ribbons);
        resize(&mut ribbons, ribbons_count);
        for (index, ribbon_point) in ribbons.iter_mut().enumerate() {
            reader.decode_ribbon_point(ribbons_start + index * RIBBON_FLOATS, ribbon_point)?;
        }
        let mut haptics = std::mem::take(&mut frame.haptics);
        resize(&mut haptics, haptics_count);
        for (index, haptic) in haptics.iter_mut().enumerate() {
            reader.decode_haptic(haptics_start + index * HAPTIC_FLOATS, haptic)?;
        }
        let mut reverb_zones = std::mem::take(&mut frame.reverb_zones);
        resize(&mut reverb_zones, reverb_zones_count);
        for (index, reverb_zone) in reverb_zones.iter_mut().enumerate() {
            reader.decode_reverb_zone(reverb_zones_start + index * REVERB_ZONE_FLOATS, reverb_zone)?;
        }
        let mut synths = std::mem::take(&mut frame.synths);
        resize(&mut synths, synths_count);
        for (index, synth_voice) in synths.iter_mut().enumerate() {
            reader.decode_synth_voice(synths_start + index * SYNTH_VOICE_FLOATS, synth_voice)?;
        }
        let mut area_lights = std::mem::take(&mut frame.area_lights);
        resize(&mut area_lights, area_lights_count);
        for (index, area_light_spec) in area_lights.iter_mut().enumerate() {
            reader.decode_area_light_spec(area_lights_start + index * AREA_LIGHT_FLOATS, area_light_spec)?;
        }
        let mut pbr_decals = std::mem::take(&mut frame.pbr_decals);
        resize(&mut pbr_decals, pbr_decals_count);
        for (index, pbr_decal_spec) in pbr_decals.iter_mut().enumerate() {
            reader.decode_pbr_decal_spec(pbr_decals_start + index * PBR_DECAL_FLOATS, pbr_decal_spec)?;
        }
        let mut camera_targets = std::mem::take(&mut frame.camera_targets);
        resize(&mut camera_targets, camera_targets_count);
        for (index, camera_target_spec) in camera_targets.iter_mut().enumerate() {
            reader.decode_camera_target_spec(camera_targets_start + index * CAMERA_TARGET_FLOATS, camera_target_spec)?;
        }
        let mut video_streams = std::mem::take(&mut frame.video_streams);
        resize(&mut video_streams, video_streams_count);
        for (index, video_stream_spec) in video_streams.iter_mut().enumerate() {
            reader.decode_video_stream_spec(video_streams_start + index * VIDEO_STREAM_FLOATS, video_stream_spec)?;
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
            quality: Quality { msaa: quality_msaa, fxaa: quality_fxaa, scale: quality_scale, filtering: quality_filtering as u32, occlusion: quality_occlusion, sky: quality_sky, effects: quality_effects, bloom: quality_bloom, taa: quality_taa, ssr: quality_ssr as u32, volumetrics: quality_volumetrics as u32, ssgi: quality_ssgi, oit: quality_oit, physical_sky: quality_physical_sky, upscale: quality_upscale, ray_tracing: quality_ray_tracing as u32, ddgi: quality_ddgi },
            sky: Sky { zenith: sky_zenith, horizon: sky_horizon, ground: sky_ground, sun: sky_sun, night: sky_night, tint: sky_tint, sh0: sky_sh0, sh1: sky_sh1, sh2: sky_sh2, sh3: sky_sh3, sh4: sky_sh4, sh5: sky_sh5, sh6: sky_sh6, sh7: sky_sh7, sh8: sky_sh8 },
            probe_volume: ProbeVolume { revision: probe_volume_revision as u32, nx: probe_volume_nx as u32, ny: probe_volume_ny as u32, nz: probe_volume_nz as u32, spacing: probe_volume_spacing, origin: probe_volume_origin },
            fog: Fog { radius: fog_radius, density: fog_density, height: fog_height, edge: fog_edge, distance: fog_distance, albedo: fog_albedo, anisotropy: fog_anisotropy, sun: fog_sun, ambient: fog_ambient, sky: fog_sky, noise: fog_noise, noise_scale: fog_noise_scale, drift: fog_drift },
            shadow: Shadow { view: shadow_view, texels: shadow_texels as u32, filter: shadow_filter as u32, softness: shadow_softness, contact: shadow_contact as u32, contact_length: shadow_contact_length, local: shadow_local as u32, local_texels: shadow_local_texels as u32 },
            blur,
            clouds: Clouds { cover: clouds_cover, drift: clouds_drift },
            hide_cursor,
            loading,
            show_captions,
            retained_enabled,
            retained_reset,
            retained_after: retained_after as u32,
            exposure: Exposure { auto: exposure_auto, compensation: exposure_compensation, min: exposure_min, max: exposure_max, brighten: exposure_brighten, darken: exposure_darken, low: exposure_low, high: exposure_high, centre: exposure_centre, range_low: exposure_range_low, range_high: exposure_range_high, target: exposure_target, dt: exposure_dt, cut: exposure_cut },
            grade: Grade { temperature: grade_temperature, tint: grade_tint, exposure: grade_exposure, contrast: grade_contrast, pivot: grade_pivot, saturation: grade_saturation, vibrance: grade_vibrance, lift: grade_lift, gamma: grade_gamma, gain: grade_gain, lut_a: grade_lut_a as u32, lut_b: grade_lut_b as u32, lut_blend: grade_lut_blend, lut_weight: grade_lut_weight, vignette: grade_vignette, vignette_radius: grade_vignette_radius, tonemap: grade_tonemap as u32, sharpness: grade_sharpness, grain: grade_grain, flare: grade_flare, outline_strength: grade_outline_strength },
            reflection: Reflection { budget: reflection_budget as u32 },
            motion_blur: MotionBlur { strength: motion_blur_strength, max_radius: motion_blur_max_radius, samples: motion_blur_samples as u32 },
            dof: Dof { focus_distance: dof_focus_distance, focus_range: dof_focus_range, aperture: dof_aperture, max_blur: dof_max_blur },
            reverb: ReverbRoom { decay: reverb_decay, damping: reverb_damping },
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
            poses,
            emitters,
            lights,
            cascades,
            probes,
            reflection_probes,
            fog_volumes,
            listener,
            panners,
            instances,
            instance_removals,
            bone_matrices,
            morph_weights,
            motions,
            ribbons,
            haptics,
            reverb_zones,
            synths,
            area_lights,
            pbr_decals,
            camera_targets,
            video_streams,
            extensions,
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

    #[cfg_attr(not(feature = "diagnostics"), allow(unused_imports))]
    use crate::diagnostic::{Actual, Check, FrameDiagnostic, Inspector, Location};
    #[cfg(feature = "diagnostics")]
    #[allow(unused_variables)]
    fn inspect_draw(d: &Inspector<'_>, catalog: &Catalog, base: usize, path: &Location, extras_cursor: &mut usize) -> Check {
        d.reference(path, "mesh", base, catalog.meshes.len(), true)?;
        d.reference(path, "material", base + 1, catalog.materials.len(), true)?;
        d.values(path, "transform", base + 2, 16, &[])?;
        d.optional(path, "parameters", base + 18, 4)?;
        d.tail(path, "extra", base + 23, MAX_DRAW_EXTRA, 4, extras_cursor)?;
        Ok(())
    }
    #[cfg(feature = "diagnostics")]
    #[allow(unused_variables)]
    fn inspect_solid_face(d: &Inspector<'_>, catalog: &Catalog, base: usize, path: &Location, extras_cursor: &mut usize) -> Check {
        d.values(path, "normal", base, 3, &["x","y","z"])?;
        d.finite(path, "distance", base + 3)?;
        d.reference(path, "material", base + 4, catalog.materials.len(), true)?;
        Ok(())
    }
    #[cfg(feature = "diagnostics")]
    #[allow(unused_variables)]
    pub(crate) fn inspect_frame(packet: &[f32], schema: u32, catalog: &Catalog) -> Check {
        let d = &Inspector { packet };
        let header = &Location::section("header");
        d.equal(header, "magic", 0, PACKET_MAGIC)?;
        d.equal(header, "schema", 1, schema as f32)?;
        let state_lanes = d.count(header, "state_count", 2, MAX_STATE_FLOATS)?;
        let draws_count = d.count(header, "draw_count", 3, MAX_DRAWS)?;
        let solids_count = d.count(header, "solid_count", 4, MAX_SOLIDS)?;
        let solids_lanes = d.count(header, "solid_lanes", 5, 32100000)?;
        let decals_count = d.count(header, "decal_count", 6, MAX_DECALS)?;
        let sounds_count = d.count(header, "sound_count", 7, MAX_SOUNDS)?;
        let sprites_count = d.count(header, "sprite_count", 8, MAX_SPRITES)?;
        let insets_count = d.count(header, "inset_count", 9, MAX_INSETS)?;
        let inset_draws_count = d.count(header, "inset_draw_count", 10, MAX_DRAWS)?;
        let preferences_lanes = d.count(header, "preference_count", 11, 256)?;
        let fields_count = d.count(header, "field_count", 157, MAX_FIELD_PATCHES)?;
        let fields_lanes = d.count(header, "field_lanes", 158, 16805888)?;
        let ui_count = d.count(header, "ui_count", 159, MAX_UI_NODES)?;
        let ui_lanes = d.count(header, "ui_lanes", 160, 4292608)?;
        let services_count = d.count(header, "service_count", 161, MAX_SERVICE_REQUESTS)?;
        let services_lanes = d.count(header, "service_lanes", 162, 4196352)?;
        let buses_count = d.count(header, "bus_count", 163, MAX_BUSES)?;
        let voices_count = d.count(header, "voice_count", 164, MAX_VOICES)?;
        let poses_count = d.count(header, "pose_count", 165, MAX_POSES)?;
        let emitters_count = d.count(header, "emitter_count", 166, MAX_EMITTERS)?;
        let lights_count = d.count(header, "light_count", 168, MAX_LIGHTS)?;
        let cascades_count = d.count(header, "cascade_count", 169, MAX_CASCADES)?;
        let probes_count = d.count(header, "probe_count", 170, MAX_PROBES)?;
        let reflection_probes_count = d.count(header, "reflection_probe_count", 171, MAX_REFLECTION_PROBES)?;
        let listener_count = d.count(header, "listener_count", 172, 1)?;
        let panners_count = d.count(header, "panner_count", 173, MAX_PANNERS)?;
        let instances_count = d.count(header, "instance_count", 179, MAX_INSTANCES)?;
        let instance_removals_count = d.count(header, "instance_removal_count", 180, MAX_INSTANCE_REMOVALS)?;
        let fog_volumes_count = d.count(header, "fog_volume_count", 234, MAX_FOG_VOLUMES)?;
        let bone_matrices_count = d.count(header, "bone_matrix_count", 235, MAX_BONE_MATRICES)?;
        let morph_weights_count = d.count(header, "morph_weight_count", 236, MAX_MORPH_WEIGHTS)?;
        let motions_count = d.count(header, "motion_count", 244, MAX_MOTIONS)?;
        let ribbons_count = d.count(header, "ribbon_count", 246, MAX_RIBBONS)?;
        let haptics_count = d.count(header, "haptic_count", 249, MAX_HAPTICS)?;
        let reverb_zones_count = d.count(header, "reverb_zone_count", 250, MAX_REVERB_ZONES)?;
        let synths_count = d.count(header, "synth_count", 251, MAX_SYNTHS)?;
        let area_lights_count = d.count(header, "area_light_count", 262, MAX_AREA_LIGHTS)?;
        let pbr_decals_count = d.count(header, "pbr_decal_count", 263, MAX_PBR_DECALS)?;
        let camera_targets_count = d.count(header, "camera_target_count", 264, MAX_CAMERA_TARGETS)?;
        let video_streams_count = d.count(header, "video_stream_count", 265, MAX_VIDEO_STREAMS)?;
        let state_start = PACKET_HEADER_FLOATS;
        let state_end = d.end(&Location::section("state"), "", state_start, state_lanes, 1)?;
        let draws_start = state_end;
        let draws_end = d.end(&Location::section("draws"), "", draws_start, draws_count, DRAW_FLOATS)?;
        let solids_start = draws_end;
        let solids_end = d.end(&Location::section("solids"), "", solids_start, solids_lanes, 1)?;
        let decals_start = solids_end;
        let decals_end = d.end(&Location::section("decals"), "", decals_start, decals_count, DECAL_FLOATS)?;
        let sounds_start = decals_end;
        let sounds_end = d.end(&Location::section("sounds"), "", sounds_start, sounds_count, SOUND_FLOATS)?;
        let sprites_start = sounds_end;
        let sprites_end = d.end(&Location::section("sprites"), "", sprites_start, sprites_count, SPRITE_FLOATS)?;
        let insets_start = sprites_end;
        let insets_end = d.end(&Location::section("insets"), "", insets_start, insets_count, INSET_FLOATS)?;
        let inset_draws_start = insets_end;
        let inset_draws_end = d.end(&Location::section("inset_draws"), "", inset_draws_start, inset_draws_count, DRAW_FLOATS)?;
        let mut inset_draws_cursor = inset_draws_start;
        let preferences_start = inset_draws_end;
        let preferences_end = d.end(&Location::section("preferences"), "", preferences_start, preferences_lanes, 1)?;
        let fields_start = preferences_end;
        let fields_end = d.end(&Location::section("fields"), "", fields_start, fields_lanes, 1)?;
        let ui_start = fields_end;
        let ui_end = d.end(&Location::section("ui"), "", ui_start, ui_lanes, 1)?;
        let services_start = ui_end;
        let services_end = d.end(&Location::section("services"), "", services_start, services_lanes, 1)?;
        let buses_start = services_end;
        let buses_end = d.end(&Location::section("buses"), "", buses_start, buses_count, BUS_FLOATS)?;
        let voices_start = buses_end;
        let voices_end = d.end(&Location::section("voices"), "", voices_start, voices_count, VOICE_FLOATS)?;
        let poses_start = voices_end;
        let poses_end = d.end(&Location::section("poses"), "", poses_start, poses_count, POSE_FLOATS)?;
        let emitters_start = poses_end;
        let emitters_end = d.end(&Location::section("emitters"), "", emitters_start, emitters_count, EMITTER_FLOATS)?;
        let lights_start = emitters_end;
        let lights_end = d.end(&Location::section("lights"), "", lights_start, lights_count, LIGHT_FLOATS)?;
        let cascades_start = lights_end;
        let cascades_end = d.end(&Location::section("cascades"), "", cascades_start, cascades_count, CASCADE_FLOATS)?;
        let probes_start = cascades_end;
        let probes_end = d.end(&Location::section("probes"), "", probes_start, probes_count, PROBE_FLOATS)?;
        let reflection_probes_start = probes_end;
        let reflection_probes_end = d.end(&Location::section("reflection_probes"), "", reflection_probes_start, reflection_probes_count, REFLECTION_PROBE_FLOATS)?;
        let fog_volumes_start = reflection_probes_end;
        let fog_volumes_end = d.end(&Location::section("fog_volumes"), "", fog_volumes_start, fog_volumes_count, FOG_VOLUME_FLOATS)?;
        let listener_start = fog_volumes_end;
        let listener_end = d.end(&Location::section("listener"), "", listener_start, listener_count, LISTENER_FLOATS)?;
        let panners_start = listener_end;
        let panners_end = d.end(&Location::section("panners"), "", panners_start, panners_count, PANNER_FLOATS)?;
        let instances_start = panners_end;
        let instances_end = d.end(&Location::section("instances"), "", instances_start, instances_count, INSTANCE_FLOATS)?;
        let instance_removals_start = instances_end;
        let instance_removals_end = d.end(&Location::section("instance_removals"), "", instance_removals_start, instance_removals_count, INSTANCE_REMOVAL_FLOATS)?;
        let bone_matrices_start = instance_removals_end;
        let bone_matrices_end = d.end(&Location::section("bone_matrices"), "", bone_matrices_start, bone_matrices_count, BONE_MATRIX_FLOATS)?;
        let morph_weights_start = bone_matrices_end;
        let morph_weights_end = d.end(&Location::section("morph_weights"), "", morph_weights_start, morph_weights_count, MORPH_WEIGHT_FLOATS)?;
        let motions_start = morph_weights_end;
        let motions_end = d.end(&Location::section("motions"), "", motions_start, motions_count, MOTION_FLOATS)?;
        let ribbons_start = motions_end;
        let ribbons_end = d.end(&Location::section("ribbons"), "", ribbons_start, ribbons_count, RIBBON_FLOATS)?;
        let haptics_start = ribbons_end;
        let haptics_end = d.end(&Location::section("haptics"), "", haptics_start, haptics_count, HAPTIC_FLOATS)?;
        let reverb_zones_start = haptics_end;
        let reverb_zones_end = d.end(&Location::section("reverb_zones"), "", reverb_zones_start, reverb_zones_count, REVERB_ZONE_FLOATS)?;
        let synths_start = reverb_zones_end;
        let synths_end = d.end(&Location::section("synths"), "", synths_start, synths_count, SYNTH_VOICE_FLOATS)?;
        let area_lights_start = synths_end;
        let area_lights_end = d.end(&Location::section("area_lights"), "", area_lights_start, area_lights_count, AREA_LIGHT_FLOATS)?;
        let pbr_decals_start = area_lights_end;
        let pbr_decals_end = d.end(&Location::section("pbr_decals"), "", pbr_decals_start, pbr_decals_count, PBR_DECAL_FLOATS)?;
        let camera_targets_start = pbr_decals_end;
        let camera_targets_end = d.end(&Location::section("camera_targets"), "", camera_targets_start, camera_targets_count, CAMERA_TARGET_FLOATS)?;
        let video_streams_start = camera_targets_end;
        let video_streams_end = d.end(&Location::section("video_streams"), "", video_streams_start, video_streams_count, VIDEO_STREAM_FLOATS)?;
        let extras_start = video_streams_end;
        let extras_cursor = &mut { extras_start };
        d.count(header, "save_requested", 12, 1)?;
        d.count(header, "load_requested", 13, 1)?;
        d.count(header, "scene_save_requested", 14, 1)?;
        d.count(header, "scene_load_requested", 15, 1)?;
        d.count(header, "checkpoint_requested", 16, 1)?;
        d.count(header, "undo_requested", 17, 1)?;
        d.count(header, "redo_requested", 18, 1)?;
        d.count(header, "exit_requested", 19, 1)?;
        d.values(header, "view", 20, 16, &[])?;
        d.values(header, "projection", 36, 16, &[])?;
        d.values(header, "clear", 52, 4, &["r","g","b","a"])?;
        d.values(header, "ambient", 56, 3, &["x","y","z"])?;
        d.values(header, "light_direction", 59, 3, &["x","y","z"])?;
        d.values(header, "light_color", 62, 3, &["x","y","z"])?;
        d.values(header, "aberration", 65, 2, &["x","y"])?;
        d.finite(header, "quality.msaa", 67)?;
        d.count(header, "quality.fxaa", 68, 1)?;
        d.finite(header, "quality.scale", 69)?;
        let _ = d.count(header, "quality.filtering", 70, 16)?;
        d.count(header, "quality.occlusion", 71, 1)?;
        d.count(header, "quality.sky", 72, 1)?;
        d.count(header, "quality.effects", 73, 1)?;
        d.values(header, "sky.zenith", 74, 3, &["x","y","z"])?;
        d.values(header, "sky.horizon", 77, 3, &["x","y","z"])?;
        d.values(header, "sky.ground", 80, 3, &["x","y","z"])?;
        d.values(header, "sky.sun", 83, 3, &["x","y","z"])?;
        d.finite(header, "sky.night", 86)?;
        d.values(header, "sky.tint", 87, 3, &["x","y","z"])?;
        d.values(header, "sky.sh0", 90, 3, &["x","y","z"])?;
        d.values(header, "sky.sh1", 93, 3, &["x","y","z"])?;
        d.values(header, "sky.sh2", 96, 3, &["x","y","z"])?;
        d.values(header, "sky.sh3", 99, 3, &["x","y","z"])?;
        d.values(header, "sky.sh4", 102, 3, &["x","y","z"])?;
        d.values(header, "sky.sh5", 105, 3, &["x","y","z"])?;
        d.values(header, "sky.sh6", 108, 3, &["x","y","z"])?;
        d.values(header, "sky.sh7", 111, 3, &["x","y","z"])?;
        d.values(header, "sky.sh8", 114, 3, &["x","y","z"])?;
        let _ = d.count(header, "probe_volume.revision", 117, 16777215)?;
        let _ = d.count(header, "probe_volume.nx", 118, 64)?;
        let _ = d.count(header, "probe_volume.ny", 119, 64)?;
        let _ = d.count(header, "probe_volume.nz", 120, 64)?;
        d.finite(header, "probe_volume.spacing", 121)?;
        d.values(header, "probe_volume.origin", 122, 3, &["x","y","z"])?;
        d.finite(header, "fog.radius", 125)?;
        d.finite(header, "fog.density", 126)?;
        d.finite(header, "fog.height", 127)?;
        d.values(header, "shadow.view", 128, 16, &[])?;
        let _ = d.count(header, "shadow.texels", 144, 8192)?;
        let _ = d.count(header, "shadow.filter", 145, 2)?;
        d.finite(header, "shadow.softness", 146)?;
        let _ = d.count(header, "shadow.contact", 147, 64)?;
        d.finite(header, "shadow.contact_length", 148)?;
        let _ = d.count(header, "shadow.local", 149, 16)?;
        let _ = d.count(header, "shadow.local_texels", 150, 4096)?;
        d.finite(header, "blur", 151)?;
        d.count(header, "quality.bloom", 152, 1)?;
        d.finite(header, "clouds.cover", 153)?;
        d.values(header, "clouds.drift", 154, 2, &["x","z"])?;
        d.count(header, "hide_cursor", 156, 1)?;
        d.count(header, "quality.taa", 167, 1)?;
        d.count(header, "loading", 174, 1)?;
        d.count(header, "show_captions", 175, 1)?;
        d.count(header, "retained_enabled", 176, 1)?;
        d.count(header, "retained_reset", 177, 1)?;
        let _ = d.count(header, "retained_after", 178, 100000)?;
        d.count(header, "exposure.auto", 181, 1)?;
        d.finite(header, "exposure.compensation", 182)?;
        d.finite(header, "exposure.min", 183)?;
        d.finite(header, "exposure.max", 184)?;
        d.finite(header, "exposure.brighten", 185)?;
        d.finite(header, "exposure.darken", 186)?;
        d.finite(header, "exposure.low", 187)?;
        d.finite(header, "exposure.high", 188)?;
        d.finite(header, "exposure.centre", 189)?;
        d.finite(header, "exposure.range_low", 190)?;
        d.finite(header, "exposure.range_high", 191)?;
        d.finite(header, "exposure.target", 192)?;
        d.finite(header, "exposure.dt", 193)?;
        d.count(header, "exposure.cut", 194, 1)?;
        d.finite(header, "grade.temperature", 195)?;
        d.finite(header, "grade.tint", 196)?;
        d.finite(header, "grade.exposure", 197)?;
        d.finite(header, "grade.contrast", 198)?;
        d.finite(header, "grade.pivot", 199)?;
        d.finite(header, "grade.saturation", 200)?;
        d.finite(header, "grade.vibrance", 201)?;
        d.values(header, "grade.lift", 202, 3, &["x","y","z"])?;
        d.values(header, "grade.gamma", 205, 3, &["x","y","z"])?;
        d.values(header, "grade.gain", 208, 3, &["x","y","z"])?;
        let _ = d.count(header, "grade.lut_a", 211, 16777215)?;
        let _ = d.count(header, "grade.lut_b", 212, 16777215)?;
        d.finite(header, "grade.lut_blend", 213)?;
        d.finite(header, "grade.lut_weight", 214)?;
        d.finite(header, "grade.vignette", 215)?;
        d.finite(header, "grade.vignette_radius", 216)?;
        let _ = d.count(header, "quality.ssr", 217, 2)?;
        let _ = d.count(header, "reflection.budget", 218, 6)?;
        d.finite(header, "fog.edge", 219)?;
        d.finite(header, "fog.distance", 220)?;
        d.values(header, "fog.albedo", 221, 3, &["x","y","z"])?;
        d.finite(header, "fog.anisotropy", 224)?;
        d.finite(header, "fog.sun", 225)?;
        d.finite(header, "fog.ambient", 226)?;
        d.finite(header, "fog.sky", 227)?;
        d.finite(header, "fog.noise", 228)?;
        d.finite(header, "fog.noise_scale", 229)?;
        d.values(header, "fog.drift", 230, 3, &["x","y","z"])?;
        let _ = d.count(header, "quality.volumetrics", 233, 2)?;
        d.finite(header, "motion_blur.strength", 237)?;
        d.finite(header, "motion_blur.max_radius", 238)?;
        let _ = d.count(header, "motion_blur.samples", 239, 32)?;
        d.finite(header, "dof.focus_distance", 240)?;
        d.finite(header, "dof.focus_range", 241)?;
        d.finite(header, "dof.aperture", 242)?;
        d.finite(header, "dof.max_blur", 243)?;
        d.count(header, "quality.ssgi", 245, 1)?;
        d.finite(header, "reverb.decay", 247)?;
        d.finite(header, "reverb.damping", 248)?;
        let _ = d.count(header, "grade.tonemap", 252, 3)?;
        d.finite(header, "grade.sharpness", 253)?;
        d.finite(header, "grade.grain", 254)?;
        d.finite(header, "grade.flare", 255)?;
        d.finite(header, "grade.outline_strength", 256)?;
        d.count(header, "quality.oit", 257, 1)?;
        d.count(header, "quality.physical_sky", 258, 1)?;
        d.count(header, "quality.upscale", 259, 1)?;
        let _ = d.count(header, "quality.ray_tracing", 260, 3)?;
        d.count(header, "quality.ddgi", 261, 1)?;
        d.values(&Location::section("state"), "", state_start, state_lanes, &[])?;
        let mut base = draws_start;
        let section_loc = &Location::section("draws");
        for item in 0..draws_count {
            let path = &section_loc.index(item);
            d.reference(path, "mesh", base, catalog.meshes.len(), true)?;
            d.reference(path, "material", base + 1, catalog.materials.len(), true)?;
            d.values(path, "transform", base + 2, 16, &[])?;
            d.optional(path, "parameters", base + 18, 4)?;
            d.tail(path, "extra", base + 23, MAX_DRAW_EXTRA, 4, extras_cursor)?;
            base += DRAW_FLOATS;
        }
        d.exhausted(section_loc, "", base, draws_end)?;
        let mut base = solids_start;
        let section_loc = &Location::section("solids");
        for item in 0..solids_count {
            let path = &section_loc.index(item);
            d.within(path, "", base, d.end(path, "", base, 1, 1)?, solids_end)?;
            let faces_count = d.min_count(path, "faces.count", base, 4, MAX_SOLID_FACES)?;
            let first = base + 1;
            let end = d.end(path, "faces", first, faces_count, SOLID_FACE_FLOATS)?;
            d.within(path, "faces", first, end, solids_end)?;
            for child in 0..faces_count {
                inspect_solid_face(d, catalog, first + child * SOLID_FACE_FLOATS, &path.field("faces").index(child), extras_cursor)?;
            }
            base = end;
        }
        d.exhausted(section_loc, "", base, solids_end)?;
        let mut base = decals_start;
        let section_loc = &Location::section("decals");
        for item in 0..decals_count {
            let path = &section_loc.index(item);
            d.reference(path, "material", base, catalog.materials.len(), true)?;
            d.values(path, "transform", base + 1, 16, &[])?;
            d.finite(path, "opacity", base + 17)?;
            base += DECAL_FLOATS;
        }
        d.exhausted(section_loc, "", base, decals_end)?;
        let mut base = sounds_start;
        let section_loc = &Location::section("sounds");
        for item in 0..sounds_count {
            let path = &section_loc.index(item);
            d.reference(path, "clip", base, catalog.sounds.len(), true)?;
            let bus = d.count(path, "bus", base + 1, 15)?;
            d.finite(path, "gain", base + 2)?;
            d.finite(path, "pan", base + 3)?;
            d.finite(path, "pitch", base + 4)?;
            d.finite(path, "lowpass", base + 5)?;
            d.finite(path, "reverb", base + 6)?;
            base += SOUND_FLOATS;
        }
        d.exhausted(section_loc, "", base, sounds_end)?;
        let mut base = sprites_start;
        let section_loc = &Location::section("sprites");
        for item in 0..sprites_count {
            let path = &section_loc.index(item);
            d.finite(path, "x", base)?;
            d.finite(path, "y", base + 1)?;
            d.finite(path, "width", base + 2)?;
            d.finite(path, "height", base + 3)?;
            d.finite(path, "u0", base + 4)?;
            d.finite(path, "v0", base + 5)?;
            d.finite(path, "u1", base + 6)?;
            d.finite(path, "v1", base + 7)?;
            d.values(path, "color", base + 8, 4, &[])?;
            d.reference(path, "material", base + 12, catalog.materials.len(), true)?;
            d.values(path, "parameters", base + 13, 4, &[])?;
            base += SPRITE_FLOATS;
        }
        d.exhausted(section_loc, "", base, sprites_end)?;
        let mut base = insets_start;
        let section_loc = &Location::section("insets");
        for item in 0..insets_count {
            let path = &section_loc.index(item);
            d.finite(path, "x", base)?;
            d.finite(path, "y", base + 1)?;
            d.finite(path, "width", base + 2)?;
            d.finite(path, "height", base + 3)?;
            d.values(path, "view", base + 4, 16, &[])?;
            d.values(path, "projection", base + 20, 16, &[])?;
            d.values(path, "ambient", base + 36, 3, &["x","y","z"])?;
            d.values(path, "light_direction", base + 39, 3, &["x","y","z"])?;
            d.values(path, "light_color", base + 42, 3, &["x","y","z"])?;
            let count = d.count(path, "draws", base + 45, MAX_DRAWS)?;
            let end = d.end(path, "draws", inset_draws_cursor, count, DRAW_FLOATS)?;
            d.within(path, "draws", inset_draws_cursor, end, inset_draws_end)?;
            for child in 0..count {
                inspect_draw(d, catalog, inset_draws_cursor, &path.field("draws").index(child), extras_cursor)?;
                inset_draws_cursor += DRAW_FLOATS;
            }
            d.finite(path, "pixel_size", base + 46)?;
            let after_sprites = d.count(path, "after_sprites", base + 47, MAX_SPRITES)?;
            base += INSET_FLOATS;
        }
        d.exhausted(section_loc, "", base, insets_end)?;
        d.exhausted(&Location::section("inset_draws"), "", inset_draws_cursor, inset_draws_end)?;
        d.values(&Location::section("preferences"), "", preferences_start, preferences_lanes, &[])?;
        let mut base = fields_start;
        let section_loc = &Location::section("fields");
        for item in 0..fields_count {
            let path = &section_loc.index(item);
            d.within(path, "", base, d.end(path, "", base, 1, 7)?, fields_end)?;
            d.reference(path, "material", base, catalog.materials.len(), true)?;
            let width = d.count(path, "width", base + 1, MAX_FIELD)?;
            let height = d.count(path, "height", base + 2, MAX_FIELD)?;
            let x = d.count(path, "x", base + 3, width)?;
            let z = d.count(path, "z", base + 4, height)?;
            let columns = d.count(path, "columns", base + 5, width - x)?;
            let rows = d.count(path, "rows", base + 6, height - z)?;
            let first = base + 7;
            let end = d.end(path, "values", first, columns * rows, 1)?;
            d.within(path, "values", first, end, fields_end)?;
            d.values(path, "values", first, end - first, &[])?;
            base = end;
        }
        d.exhausted(section_loc, "", base, fields_end)?;
        let mut base = ui_start;
        let section_loc = &Location::section("ui");
        for item in 0..ui_count {
            let path = &section_loc.index(item);
            d.within(path, "", base, d.end(path, "", base, 1, 24)?, ui_end)?;
            let id = d.count(path, "id", base, MAX_UI_ID)?;
            let parent = d.count(path, "parent", base + 1, MAX_UI_NODES)?;
            let role = d.count(path, "role", base + 2, 23)?;
            d.finite(path, "x", base + 3)?;
            d.finite(path, "y", base + 4)?;
            d.finite(path, "width", base + 5)?;
            d.finite(path, "height", base + 6)?;
            d.count(path, "focusable", base + 7, 1)?;
            d.count(path, "focused", base + 8, 1)?;
            d.count(path, "disabled", base + 9, 1)?;
            d.count(path, "hidden", base + 10, 1)?;
            d.count(path, "modal", base + 11, 1)?;
            let checked = d.count(path, "checked", base + 12, 3)?;
            let expanded = d.count(path, "expanded", base + 13, 2)?;
            let selected = d.count(path, "selected", base + 14, 2)?;
            let live = d.count(path, "live", base + 15, 2)?;
            d.finite(path, "value", base + 16)?;
            d.finite(path, "minimum", base + 17)?;
            d.finite(path, "maximum", base + 18)?;
            d.finite(path, "step", base + 19)?;
            let actions = d.count(path, "actions", base + 20, 255)?;
            let text_length = d.count(path, "text_length", base + 21, MAX_UI_TEXT)?;
            let name_length = d.count(path, "name_length", base + 22, text_length)?;
            let description_length = d.count(path, "description_length", base + 23, text_length - name_length)?;
            let first = base + 24;
            let end = d.end(path, "text", first, text_length, 1)?;
            d.within(path, "text", first, end, ui_end)?;
            d.values(path, "text", first, end - first, &[])?;
            base = end;
        }
        d.exhausted(section_loc, "", base, ui_end)?;
        let mut base = services_start;
        let section_loc = &Location::section("services");
        for item in 0..services_count {
            let path = &section_loc.index(item);
            d.within(path, "", base, d.end(path, "", base, 1, 8)?, services_end)?;
            let request = d.count(path, "request", base, MAX_SERVICE_ID)?;
            let service = d.count(path, "service", base + 1, 15)?;
            let operation = d.count(path, "operation", base + 2, MAX_SERVICE_OPERATION)?;
            d.finite(path, "a", base + 3)?;
            d.finite(path, "b", base + 4)?;
            d.finite(path, "c", base + 5)?;
            d.finite(path, "d", base + 6)?;
            let value_count = d.count(path, "value_count", base + 7, MAX_SERVICE_DATA)?;
            let first = base + 8;
            let end = d.end(path, "values", first, value_count, 1)?;
            d.within(path, "values", first, end, services_end)?;
            d.values(path, "values", first, end - first, &[])?;
            base = end;
        }
        d.exhausted(section_loc, "", base, services_end)?;
        let mut base = buses_start;
        let section_loc = &Location::section("buses");
        for item in 0..buses_count {
            let path = &section_loc.index(item);
            d.finite(path, "gain", base)?;
            d.finite(path, "lowpass", base + 1)?;
            d.finite(path, "reverb", base + 2)?;
            d.finite(path, "comp_threshold", base + 3)?;
            d.finite(path, "comp_ratio", base + 4)?;
            let duck_bus = d.count(path, "duck_bus", base + 5, 15)?;
            d.finite(path, "duck_amount", base + 6)?;
            d.finite(path, "eq_low_db", base + 7)?;
            d.finite(path, "eq_mid_db", base + 8)?;
            d.finite(path, "eq_high_db", base + 9)?;
            base += BUS_FLOATS;
        }
        d.exhausted(section_loc, "", base, buses_end)?;
        let mut base = voices_start;
        let section_loc = &Location::section("voices");
        for item in 0..voices_count {
            let path = &section_loc.index(item);
            let id = d.count(path, "id", base, MAX_VOICE_ID)?;
            d.reference(path, "clip", base + 1, catalog.sounds.len(), true)?;
            let bus = d.count(path, "bus", base + 2, 15)?;
            d.finite(path, "gain", base + 3)?;
            d.finite(path, "pan", base + 4)?;
            d.finite(path, "pitch", base + 5)?;
            d.finite(path, "lowpass", base + 6)?;
            d.finite(path, "reverb", base + 7)?;
            d.count(path, "looping", base + 8, 1)?;
            d.finite(path, "start", base + 9)?;
            d.finite(path, "fade_in", base + 10)?;
            d.finite(path, "fade_out", base + 11)?;
            base += VOICE_FLOATS;
        }
        d.exhausted(section_loc, "", base, voices_end)?;
        let mut base = poses_start;
        let section_loc = &Location::section("poses");
        for item in 0..poses_count {
            let path = &section_loc.index(item);
            let draw = d.count(path, "draw", base, MAX_DRAWS)?;
            let clip = d.count(path, "clip", base + 1, 65535)?;
            d.finite(path, "time", base + 2)?;
            d.finite(path, "weight", base + 3)?;
            let inset = d.count(path, "inset", base + 4, 16)?;
            base += POSE_FLOATS;
        }
        d.exhausted(section_loc, "", base, poses_end)?;
        let mut base = emitters_start;
        let section_loc = &Location::section("emitters");
        for item in 0..emitters_count {
            let path = &section_loc.index(item);
            d.reference(path, "material", base, catalog.materials.len(), true)?;
            d.values(path, "transform", base + 1, 16, &[])?;
            let seed = d.count(path, "seed", base + 17, 16777215)?;
            d.finite(path, "time", base + 18)?;
            d.finite(path, "duration", base + 19)?;
            d.finite(path, "rate", base + 20)?;
            d.finite(path, "lifetime", base + 21)?;
            d.finite(path, "radius", base + 22)?;
            d.values(path, "velocity", base + 23, 3, &["x","y","z"])?;
            d.finite(path, "spread", base + 26)?;
            d.values(path, "gravity", base + 27, 3, &["x","y","z"])?;
            d.finite(path, "drag", base + 30)?;
            d.finite(path, "size_start", base + 31)?;
            d.finite(path, "size_end", base + 32)?;
            d.values(path, "color_start", base + 33, 4, &["r","g","b","a"])?;
            d.values(path, "color_end", base + 37, 4, &["r","g","b","a"])?;
            d.count(path, "additive", base + 41, 1)?;
            d.finite(path, "soft_depth", base + 42)?;
            d.finite(path, "lit", base + 43)?;
            let flipbook_cols = d.count(path, "flipbook_cols", base + 44, 32)?;
            let flipbook_rows = d.count(path, "flipbook_rows", base + 45, 32)?;
            d.finite(path, "flipbook_fps", base + 46)?;
            base += EMITTER_FLOATS;
        }
        d.exhausted(section_loc, "", base, emitters_end)?;
        let mut base = lights_start;
        let section_loc = &Location::section("lights");
        for item in 0..lights_count {
            let path = &section_loc.index(item);
            d.values(path, "position", base, 3, &["x","y","z"])?;
            d.finite(path, "radius", base + 3)?;
            d.values(path, "color", base + 4, 3, &["x","y","z"])?;
            d.values(path, "direction", base + 7, 3, &["x","y","z"])?;
            d.finite(path, "inner", base + 10)?;
            d.finite(path, "outer", base + 11)?;
            d.count(path, "shadow", base + 12, 1)?;
            let id = d.count(path, "id", base + 13, 16777215)?;
            base += LIGHT_FLOATS;
        }
        d.exhausted(section_loc, "", base, lights_end)?;
        let mut base = cascades_start;
        let section_loc = &Location::section("cascades");
        for item in 0..cascades_count {
            let path = &section_loc.index(item);
            d.values(path, "projection", base, 16, &[])?;
            d.finite(path, "texel", base + 16)?;
            d.finite(path, "far", base + 17)?;
            base += CASCADE_FLOATS;
        }
        d.exhausted(section_loc, "", base, cascades_end)?;
        let mut base = probes_start;
        let section_loc = &Location::section("probes");
        for item in 0..probes_count {
            let path = &section_loc.index(item);
            d.values(path, "sh0", base, 3, &["x","y","z"])?;
            d.values(path, "sh1", base + 3, 3, &["x","y","z"])?;
            d.values(path, "sh2", base + 6, 3, &["x","y","z"])?;
            d.values(path, "sh3", base + 9, 3, &["x","y","z"])?;
            d.values(path, "sh4", base + 12, 3, &["x","y","z"])?;
            d.values(path, "sh5", base + 15, 3, &["x","y","z"])?;
            d.values(path, "sh6", base + 18, 3, &["x","y","z"])?;
            d.values(path, "sh7", base + 21, 3, &["x","y","z"])?;
            d.values(path, "sh8", base + 24, 3, &["x","y","z"])?;
            d.finite(path, "valid", base + 27)?;
            base += PROBE_FLOATS;
        }
        d.exhausted(section_loc, "", base, probes_end)?;
        let mut base = reflection_probes_start;
        let section_loc = &Location::section("reflection_probes");
        for item in 0..reflection_probes_count {
            let path = &section_loc.index(item);
            let id = d.count(path, "id", base, 16777215)?;
            let revision = d.count(path, "revision", base + 1, 16777215)?;
            let priority = d.count(path, "priority", base + 2, 255)?;
            d.count(path, "capture", base + 3, 1)?;
            d.values(path, "position", base + 4, 3, &["x","y","z"])?;
            d.values(path, "center", base + 7, 3, &["x","y","z"])?;
            d.values(path, "extent", base + 10, 3, &["x","y","z"])?;
            d.finite(path, "blend", base + 13)?;
            d.values(path, "parallax_center", base + 14, 3, &["x","y","z"])?;
            d.values(path, "parallax_extent", base + 17, 3, &["x","y","z"])?;
            d.finite(path, "far", base + 20)?;
            base += REFLECTION_PROBE_FLOATS;
        }
        d.exhausted(section_loc, "", base, reflection_probes_end)?;
        let mut base = fog_volumes_start;
        let section_loc = &Location::section("fog_volumes");
        for item in 0..fog_volumes_count {
            let path = &section_loc.index(item);
            let shape = d.count(path, "shape", base, 1)?;
            d.values(path, "center", base + 1, 3, &["x","y","z"])?;
            d.values(path, "extent", base + 4, 3, &["x","y","z"])?;
            d.finite(path, "density", base + 7)?;
            d.values(path, "albedo", base + 8, 3, &["x","y","z"])?;
            d.values(path, "emission", base + 11, 3, &["x","y","z"])?;
            d.finite(path, "falloff", base + 14)?;
            base += FOG_VOLUME_FLOATS;
        }
        d.exhausted(section_loc, "", base, fog_volumes_end)?;
        let mut base = listener_start;
        let section_loc = &Location::section("listener");
        for item in 0..listener_count {
            let path = &section_loc.index(item);
            d.values(path, "position", base, 3, &["x","y","z"])?;
            d.values(path, "forward", base + 3, 3, &["x","y","z"])?;
            d.values(path, "up", base + 6, 3, &["x","y","z"])?;
            d.values(path, "velocity", base + 9, 3, &["x","y","z"])?;
            base += LISTENER_FLOATS;
        }
        d.exhausted(section_loc, "", base, listener_end)?;
        let mut base = panners_start;
        let section_loc = &Location::section("panners");
        for item in 0..panners_count {
            let path = &section_loc.index(item);
            let voice = d.count(path, "voice", base, MAX_VOICE_ID)?;
            let sound = d.count(path, "sound", base + 1, 255)?;
            d.values(path, "position", base + 2, 3, &["x","y","z"])?;
            d.values(path, "orientation", base + 5, 3, &["x","y","z"])?;
            let distance_model = d.count(path, "distance_model", base + 8, 2)?;
            d.finite(path, "ref_distance", base + 9)?;
            d.finite(path, "max_distance", base + 10)?;
            d.finite(path, "rolloff", base + 11)?;
            d.finite(path, "cone_inner", base + 12)?;
            d.finite(path, "cone_outer", base + 13)?;
            d.finite(path, "cone_outer_gain", base + 14)?;
            d.count(path, "hrtf", base + 15, 1)?;
            d.values(path, "velocity", base + 16, 3, &["x","y","z"])?;
            d.finite(path, "doppler", base + 19)?;
            d.finite(path, "occlusion", base + 20)?;
            let zone = d.count(path, "zone", base + 21, 64)?;
            base += PANNER_FLOATS;
        }
        d.exhausted(section_loc, "", base, panners_end)?;
        let mut base = instances_start;
        let section_loc = &Location::section("instances");
        for item in 0..instances_count {
            let path = &section_loc.index(item);
            let id_low = d.count(path, "id_low", base, 65535)?;
            let id_high = d.count(path, "id_high", base + 1, 65535)?;
            let order = d.count(path, "order", base + 2, 16777215)?;
            d.reference(path, "mesh", base + 3, catalog.meshes.len(), true)?;
            d.reference(path, "material", base + 4, catalog.materials.len(), true)?;
            d.values(path, "transform", base + 5, 16, &[])?;
            d.optional(path, "parameters", base + 21, 4)?;
            base += INSTANCE_FLOATS;
        }
        d.exhausted(section_loc, "", base, instances_end)?;
        let mut base = instance_removals_start;
        let section_loc = &Location::section("instance_removals");
        for item in 0..instance_removals_count {
            let path = &section_loc.index(item);
            let id_low = d.count(path, "id_low", base, 65535)?;
            let id_high = d.count(path, "id_high", base + 1, 65535)?;
            base += INSTANCE_REMOVAL_FLOATS;
        }
        d.exhausted(section_loc, "", base, instance_removals_end)?;
        let mut base = bone_matrices_start;
        let section_loc = &Location::section("bone_matrices");
        for item in 0..bone_matrices_count {
            let path = &section_loc.index(item);
            let inset = d.count(path, "inset", base, 16)?;
            let draw = d.count(path, "draw", base + 1, MAX_DRAWS)?;
            let joint = d.count(path, "joint", base + 2, 1024)?;
            d.values(path, "matrix", base + 3, 16, &[])?;
            base += BONE_MATRIX_FLOATS;
        }
        d.exhausted(section_loc, "", base, bone_matrices_end)?;
        let mut base = morph_weights_start;
        let section_loc = &Location::section("morph_weights");
        for item in 0..morph_weights_count {
            let path = &section_loc.index(item);
            let inset = d.count(path, "inset", base, 16)?;
            let draw = d.count(path, "draw", base + 1, MAX_DRAWS)?;
            let target = d.count(path, "target", base + 2, 256)?;
            d.finite(path, "weight", base + 3)?;
            base += MORPH_WEIGHT_FLOATS;
        }
        d.exhausted(section_loc, "", base, morph_weights_end)?;
        let mut base = motions_start;
        let section_loc = &Location::section("motions");
        for item in 0..motions_count {
            let path = &section_loc.index(item);
            let draw = d.count(path, "draw", base, MAX_DRAWS)?;
            d.values(path, "previous", base + 1, 16, &[])?;
            base += MOTION_FLOATS;
        }
        d.exhausted(section_loc, "", base, motions_end)?;
        let mut base = ribbons_start;
        let section_loc = &Location::section("ribbons");
        for item in 0..ribbons_count {
            let path = &section_loc.index(item);
            let ribbon = d.count(path, "ribbon", base, 65535)?;
            d.reference(path, "material", base + 1, catalog.materials.len(), true)?;
            d.count(path, "additive", base + 2, 1)?;
            d.values(path, "position", base + 3, 3, &["x","y","z"])?;
            d.finite(path, "width", base + 6)?;
            d.values(path, "color", base + 7, 4, &["r","g","b","a"])?;
            d.finite(path, "soft_depth", base + 11)?;
            base += RIBBON_FLOATS;
        }
        d.exhausted(section_loc, "", base, ribbons_end)?;
        let mut base = haptics_start;
        let section_loc = &Location::section("haptics");
        for item in 0..haptics_count {
            let path = &section_loc.index(item);
            let pad = d.count(path, "pad", base, 15)?;
            d.finite(path, "low", base + 1)?;
            d.finite(path, "high", base + 2)?;
            d.finite(path, "duration", base + 3)?;
            base += HAPTIC_FLOATS;
        }
        d.exhausted(section_loc, "", base, haptics_end)?;
        let mut base = reverb_zones_start;
        let section_loc = &Location::section("reverb_zones");
        for item in 0..reverb_zones_count {
            let path = &section_loc.index(item);
            let id = d.count(path, "id", base, 64)?;
            d.values(path, "position", base + 1, 3, &["x","y","z"])?;
            d.values(path, "half_extents", base + 4, 3, &["x","y","z"])?;
            d.finite(path, "blend", base + 7)?;
            d.finite(path, "decay", base + 8)?;
            d.finite(path, "damping", base + 9)?;
            d.finite(path, "wet", base + 10)?;
            d.finite(path, "portal_attenuation", base + 11)?;
            base += REVERB_ZONE_FLOATS;
        }
        d.exhausted(section_loc, "", base, reverb_zones_end)?;
        let mut base = synths_start;
        let section_loc = &Location::section("synths");
        for item in 0..synths_count {
            let path = &section_loc.index(item);
            let id = d.count(path, "id", base, MAX_VOICE_ID)?;
            let mode = d.count(path, "mode", base + 1, 5)?;
            let bus = d.count(path, "bus", base + 2, 15)?;
            d.finite(path, "frequency", base + 3)?;
            d.finite(path, "gain", base + 4)?;
            d.finite(path, "pan", base + 5)?;
            d.finite(path, "detune", base + 6)?;
            d.finite(path, "attack", base + 7)?;
            d.finite(path, "decay", base + 8)?;
            d.finite(path, "sustain", base + 9)?;
            d.finite(path, "release", base + 10)?;
            d.finite(path, "cutoff", base + 11)?;
            d.finite(path, "resonance", base + 12)?;
            d.finite(path, "grain_size", base + 13)?;
            d.finite(path, "grain_pitch", base + 14)?;
            d.finite(path, "reverb", base + 15)?;
            base += SYNTH_VOICE_FLOATS;
        }
        d.exhausted(section_loc, "", base, synths_end)?;
        let mut base = area_lights_start;
        let section_loc = &Location::section("area_lights");
        for item in 0..area_lights_count {
            let path = &section_loc.index(item);
            let kind = d.count(path, "kind", base, 3)?;
            d.values(path, "position", base + 1, 3, &["x","y","z"])?;
            d.values(path, "right", base + 4, 3, &["x","y","z"])?;
            d.values(path, "up", base + 7, 3, &["x","y","z"])?;
            d.values(path, "color", base + 10, 3, &["x","y","z"])?;
            d.finite(path, "intensity", base + 13)?;
            d.finite(path, "width", base + 14)?;
            d.finite(path, "height", base + 15)?;
            d.finite(path, "range", base + 16)?;
            let cookie_material = d.count(path, "cookie_material", base + 17, 65535)?;
            d.count(path, "two_sided", base + 18, 1)?;
            base += AREA_LIGHT_FLOATS;
        }
        d.exhausted(section_loc, "", base, area_lights_end)?;
        let mut base = pbr_decals_start;
        let section_loc = &Location::section("pbr_decals");
        for item in 0..pbr_decals_count {
            let path = &section_loc.index(item);
            d.reference(path, "material", base, catalog.materials.len(), true)?;
            d.values(path, "transform", base + 1, 16, &[])?;
            d.values(path, "albedo", base + 17, 4, &["r","g","b","a"])?;
            d.finite(path, "normal_strength", base + 21)?;
            d.finite(path, "roughness", base + 22)?;
            d.finite(path, "metallic", base + 23)?;
            d.finite(path, "emissive_strength", base + 24)?;
            d.finite(path, "fade", base + 25)?;
            base += PBR_DECAL_FLOATS;
        }
        d.exhausted(section_loc, "", base, pbr_decals_end)?;
        let mut base = camera_targets_start;
        let section_loc = &Location::section("camera_targets");
        for item in 0..camera_targets_count {
            let path = &section_loc.index(item);
            d.reference(path, "material", base, catalog.materials.len(), true)?;
            let width = d.count(path, "width", base + 1, 4096)?;
            let height = d.count(path, "height", base + 2, 4096)?;
            d.finite(path, "viewport_x", base + 3)?;
            d.finite(path, "viewport_y", base + 4)?;
            d.finite(path, "viewport_w", base + 5)?;
            d.finite(path, "viewport_h", base + 6)?;
            d.count(path, "split_screen", base + 7, 1)?;
            d.values(path, "view", base + 8, 16, &[])?;
            d.values(path, "projection", base + 24, 16, &[])?;
            base += CAMERA_TARGET_FLOATS;
        }
        d.exhausted(section_loc, "", base, camera_targets_end)?;
        let mut base = video_streams_start;
        let section_loc = &Location::section("video_streams");
        for item in 0..video_streams_count {
            let path = &section_loc.index(item);
            d.reference(path, "material", base, catalog.materials.len(), true)?;
            d.finite(path, "time", base + 1)?;
            d.count(path, "playing", base + 2, 1)?;
            d.count(path, "looping", base + 3, 1)?;
            d.finite(path, "speed", base + 4)?;
            base += VIDEO_STREAM_FLOATS;
        }
        d.exhausted(section_loc, "", base, video_streams_end)?;
        inspect_extensions(d, *extras_cursor)?;
        Ok(())
    }
    #[cfg(not(feature = "diagnostics"))]
    #[cold]
    fn explain_frame(packet: &[f32], _schema: u32, _catalog: &Catalog) -> FrameDiagnostic {
        Location::section("packet").error(0, "a valid render packet", Actual::Length(packet.len()))
    }
    #[cfg(feature = "diagnostics")]
    #[cold]
    fn explain_frame(packet: &[f32], schema: u32, catalog: &Catalog) -> FrameDiagnostic {
        let mut diagnostic = inspect_frame(packet, schema, catalog).err().unwrap_or_else(|| Location::section("packet").error(0, "a valid render packet (structural context unavailable)", Actual::Length(packet.len())));
        if diagnostic.section != "extensions" && let Ok(start) = provenance_start(packet, schema) {
            diagnostic.origin = crate::diagnostic::origin(packet, start, diagnostic.lane).map(Box::new);
        }
        diagnostic
    }
    #[cfg(feature = "diagnostics")]
    #[allow(unused_variables, unused_mut)]
    fn provenance_start(packet: &[f32], schema: u32) -> Check<usize> {
        let d = &Inspector { packet };
        let header = &Location::section("header");
        d.equal(header, "magic", 0, PACKET_MAGIC)?;
        d.equal(header, "schema", 1, schema as f32)?;
        let state_lanes = d.count(header, "state_count", 2, MAX_STATE_FLOATS)?;
        let draws_count = d.count(header, "draw_count", 3, MAX_DRAWS)?;
        let solids_count = d.count(header, "solid_count", 4, MAX_SOLIDS)?;
        let solids_lanes = d.count(header, "solid_lanes", 5, 32100000)?;
        let decals_count = d.count(header, "decal_count", 6, MAX_DECALS)?;
        let sounds_count = d.count(header, "sound_count", 7, MAX_SOUNDS)?;
        let sprites_count = d.count(header, "sprite_count", 8, MAX_SPRITES)?;
        let insets_count = d.count(header, "inset_count", 9, MAX_INSETS)?;
        let inset_draws_count = d.count(header, "inset_draw_count", 10, MAX_DRAWS)?;
        let preferences_lanes = d.count(header, "preference_count", 11, 256)?;
        let fields_count = d.count(header, "field_count", 157, MAX_FIELD_PATCHES)?;
        let fields_lanes = d.count(header, "field_lanes", 158, 16805888)?;
        let ui_count = d.count(header, "ui_count", 159, MAX_UI_NODES)?;
        let ui_lanes = d.count(header, "ui_lanes", 160, 4292608)?;
        let services_count = d.count(header, "service_count", 161, MAX_SERVICE_REQUESTS)?;
        let services_lanes = d.count(header, "service_lanes", 162, 4196352)?;
        let buses_count = d.count(header, "bus_count", 163, MAX_BUSES)?;
        let voices_count = d.count(header, "voice_count", 164, MAX_VOICES)?;
        let poses_count = d.count(header, "pose_count", 165, MAX_POSES)?;
        let emitters_count = d.count(header, "emitter_count", 166, MAX_EMITTERS)?;
        let lights_count = d.count(header, "light_count", 168, MAX_LIGHTS)?;
        let cascades_count = d.count(header, "cascade_count", 169, MAX_CASCADES)?;
        let probes_count = d.count(header, "probe_count", 170, MAX_PROBES)?;
        let reflection_probes_count = d.count(header, "reflection_probe_count", 171, MAX_REFLECTION_PROBES)?;
        let listener_count = d.count(header, "listener_count", 172, 1)?;
        let panners_count = d.count(header, "panner_count", 173, MAX_PANNERS)?;
        let instances_count = d.count(header, "instance_count", 179, MAX_INSTANCES)?;
        let instance_removals_count = d.count(header, "instance_removal_count", 180, MAX_INSTANCE_REMOVALS)?;
        let fog_volumes_count = d.count(header, "fog_volume_count", 234, MAX_FOG_VOLUMES)?;
        let bone_matrices_count = d.count(header, "bone_matrix_count", 235, MAX_BONE_MATRICES)?;
        let morph_weights_count = d.count(header, "morph_weight_count", 236, MAX_MORPH_WEIGHTS)?;
        let motions_count = d.count(header, "motion_count", 244, MAX_MOTIONS)?;
        let ribbons_count = d.count(header, "ribbon_count", 246, MAX_RIBBONS)?;
        let haptics_count = d.count(header, "haptic_count", 249, MAX_HAPTICS)?;
        let reverb_zones_count = d.count(header, "reverb_zone_count", 250, MAX_REVERB_ZONES)?;
        let synths_count = d.count(header, "synth_count", 251, MAX_SYNTHS)?;
        let area_lights_count = d.count(header, "area_light_count", 262, MAX_AREA_LIGHTS)?;
        let pbr_decals_count = d.count(header, "pbr_decal_count", 263, MAX_PBR_DECALS)?;
        let camera_targets_count = d.count(header, "camera_target_count", 264, MAX_CAMERA_TARGETS)?;
        let video_streams_count = d.count(header, "video_stream_count", 265, MAX_VIDEO_STREAMS)?;
        let state_start = PACKET_HEADER_FLOATS;
        let state_end = d.end(&Location::section("state"), "", state_start, state_lanes, 1)?;
        let draws_start = state_end;
        let draws_end = d.end(&Location::section("draws"), "", draws_start, draws_count, DRAW_FLOATS)?;
        let solids_start = draws_end;
        let solids_end = d.end(&Location::section("solids"), "", solids_start, solids_lanes, 1)?;
        let decals_start = solids_end;
        let decals_end = d.end(&Location::section("decals"), "", decals_start, decals_count, DECAL_FLOATS)?;
        let sounds_start = decals_end;
        let sounds_end = d.end(&Location::section("sounds"), "", sounds_start, sounds_count, SOUND_FLOATS)?;
        let sprites_start = sounds_end;
        let sprites_end = d.end(&Location::section("sprites"), "", sprites_start, sprites_count, SPRITE_FLOATS)?;
        let insets_start = sprites_end;
        let insets_end = d.end(&Location::section("insets"), "", insets_start, insets_count, INSET_FLOATS)?;
        let inset_draws_start = insets_end;
        let inset_draws_end = d.end(&Location::section("inset_draws"), "", inset_draws_start, inset_draws_count, DRAW_FLOATS)?;
        let mut inset_draws_cursor = inset_draws_start;
        let preferences_start = inset_draws_end;
        let preferences_end = d.end(&Location::section("preferences"), "", preferences_start, preferences_lanes, 1)?;
        let fields_start = preferences_end;
        let fields_end = d.end(&Location::section("fields"), "", fields_start, fields_lanes, 1)?;
        let ui_start = fields_end;
        let ui_end = d.end(&Location::section("ui"), "", ui_start, ui_lanes, 1)?;
        let services_start = ui_end;
        let services_end = d.end(&Location::section("services"), "", services_start, services_lanes, 1)?;
        let buses_start = services_end;
        let buses_end = d.end(&Location::section("buses"), "", buses_start, buses_count, BUS_FLOATS)?;
        let voices_start = buses_end;
        let voices_end = d.end(&Location::section("voices"), "", voices_start, voices_count, VOICE_FLOATS)?;
        let poses_start = voices_end;
        let poses_end = d.end(&Location::section("poses"), "", poses_start, poses_count, POSE_FLOATS)?;
        let emitters_start = poses_end;
        let emitters_end = d.end(&Location::section("emitters"), "", emitters_start, emitters_count, EMITTER_FLOATS)?;
        let lights_start = emitters_end;
        let lights_end = d.end(&Location::section("lights"), "", lights_start, lights_count, LIGHT_FLOATS)?;
        let cascades_start = lights_end;
        let cascades_end = d.end(&Location::section("cascades"), "", cascades_start, cascades_count, CASCADE_FLOATS)?;
        let probes_start = cascades_end;
        let probes_end = d.end(&Location::section("probes"), "", probes_start, probes_count, PROBE_FLOATS)?;
        let reflection_probes_start = probes_end;
        let reflection_probes_end = d.end(&Location::section("reflection_probes"), "", reflection_probes_start, reflection_probes_count, REFLECTION_PROBE_FLOATS)?;
        let fog_volumes_start = reflection_probes_end;
        let fog_volumes_end = d.end(&Location::section("fog_volumes"), "", fog_volumes_start, fog_volumes_count, FOG_VOLUME_FLOATS)?;
        let listener_start = fog_volumes_end;
        let listener_end = d.end(&Location::section("listener"), "", listener_start, listener_count, LISTENER_FLOATS)?;
        let panners_start = listener_end;
        let panners_end = d.end(&Location::section("panners"), "", panners_start, panners_count, PANNER_FLOATS)?;
        let instances_start = panners_end;
        let instances_end = d.end(&Location::section("instances"), "", instances_start, instances_count, INSTANCE_FLOATS)?;
        let instance_removals_start = instances_end;
        let instance_removals_end = d.end(&Location::section("instance_removals"), "", instance_removals_start, instance_removals_count, INSTANCE_REMOVAL_FLOATS)?;
        let bone_matrices_start = instance_removals_end;
        let bone_matrices_end = d.end(&Location::section("bone_matrices"), "", bone_matrices_start, bone_matrices_count, BONE_MATRIX_FLOATS)?;
        let morph_weights_start = bone_matrices_end;
        let morph_weights_end = d.end(&Location::section("morph_weights"), "", morph_weights_start, morph_weights_count, MORPH_WEIGHT_FLOATS)?;
        let motions_start = morph_weights_end;
        let motions_end = d.end(&Location::section("motions"), "", motions_start, motions_count, MOTION_FLOATS)?;
        let ribbons_start = motions_end;
        let ribbons_end = d.end(&Location::section("ribbons"), "", ribbons_start, ribbons_count, RIBBON_FLOATS)?;
        let haptics_start = ribbons_end;
        let haptics_end = d.end(&Location::section("haptics"), "", haptics_start, haptics_count, HAPTIC_FLOATS)?;
        let reverb_zones_start = haptics_end;
        let reverb_zones_end = d.end(&Location::section("reverb_zones"), "", reverb_zones_start, reverb_zones_count, REVERB_ZONE_FLOATS)?;
        let synths_start = reverb_zones_end;
        let synths_end = d.end(&Location::section("synths"), "", synths_start, synths_count, SYNTH_VOICE_FLOATS)?;
        let area_lights_start = synths_end;
        let area_lights_end = d.end(&Location::section("area_lights"), "", area_lights_start, area_lights_count, AREA_LIGHT_FLOATS)?;
        let pbr_decals_start = area_lights_end;
        let pbr_decals_end = d.end(&Location::section("pbr_decals"), "", pbr_decals_start, pbr_decals_count, PBR_DECAL_FLOATS)?;
        let camera_targets_start = pbr_decals_end;
        let camera_targets_end = d.end(&Location::section("camera_targets"), "", camera_targets_start, camera_targets_count, CAMERA_TARGET_FLOATS)?;
        let video_streams_start = camera_targets_end;
        let video_streams_end = d.end(&Location::section("video_streams"), "", video_streams_start, video_streams_count, VIDEO_STREAM_FLOATS)?;
        let extras_start = video_streams_end;
        let mut end = extras_start;
        for item in 0..draws_count {
            let count = d.count(&Location::section("draws").index(item), "extra", draws_start + item * DRAW_FLOATS + 23, MAX_DRAW_EXTRA)?;
            end = d.end(&Location::section("extras"), "", end, count, 4)?;
        }
        for item in 0..inset_draws_count {
            let count = d.count(&Location::section("inset_draws").index(item), "extra", inset_draws_start + item * DRAW_FLOATS + 23, MAX_DRAW_EXTRA)?;
            end = d.end(&Location::section("extras"), "", end, count, 4)?;
        }
        Ok(end)
    }
    #[cfg(feature = "diagnostics")]
    fn inspect_extensions(d: &Inspector<'_>, mut at: usize) -> Check {
        let start = at;
        let path = &Location::section("extensions");
        let mut provenance_seen = false;
        if at < d.packet.len() {
            d.equal(path, "marker", at, EXTENSIONS_MARKER)?;
            let count = d.count(path, "count", at + 1, MAX_EXTENSIONS)?;
            at += 2;
            for item in 0..count {
                let path = &path.index(item);
                let tag = d.count(path, "tag", at, MAX_EXTENSION_TAG)?;
                let length = d.count(path, "length", at + 1, if tag == PROVENANCE_TAG { MAX_PROVENANCE_LANES } else { MAX_EXTENSION_LANES })?;
                d.values(path, "values", at + 2, length, &[])?;
                if tag == PROVENANCE_TAG {
                    if provenance_seen || crate::diagnostic::read_provenance(&d.packet[at + 2..at + 2 + length], start).is_err() {
                        return Err(path.field_err("provenance", at + 2, "one bounded version-1 provenance JSON object", Actual::Length(length)));
                    }
                    provenance_seen = true;
                }
                at = d.end(path, "", at + 2, length, 1)?;
            }
        }
        d.exhausted(path, "", at, d.packet.len())
    }

    /// The walk of an input packet's values that pen's tests compare with the game's
    /// (`pen/protocol_canonical.sq`): elapsed, the viewport's width and height, the events'
    /// count, each event's tag and then its fields in the schema's order (a key as its code),
    /// then the state's count and lanes. It lays nothing out as the packet does.
    pub fn canonical_input(frame: &Frame, elapsed: f64, state: &[f32]) -> Vec<f32> {
        let viewport = frame.viewport;
        let mut out = vec![elapsed as f32, viewport.width as f32, viewport.height as f32, frame.events.len() as f32];
        for event in &frame.events {
            match event {
                GameEvent::KeyDown(value) => out.extend([1.0, key_code(value)]),
                GameEvent::KeyUp(value) => out.extend([2.0, key_code(value)]),
                GameEvent::PointerDown { button, x, y } => out.extend([3.0, *button as f32, *x, *y]),
                GameEvent::PointerUp { button, x, y } => out.extend([4.0, *button as f32, *x, *y]),
                GameEvent::PointerMove { x, y } => out.extend([5.0, *x, *y]),
                GameEvent::Wheel { x, y, delta } => out.extend([6.0, *x, *y, *delta]),
                GameEvent::FocusLost => out.push(7.0),
                GameEvent::AssetReady { .. } => out.push(8.0),
                GameEvent::AssetFailed { .. } => out.push(8.0),
                GameEvent::Console { command, argument } => out.extend([9.0, *command, *argument]),
                GameEvent::Preference { index, value } => out.extend([10.0, *index, *value]),
                GameEvent::Tune { index, value } => out.extend([11.0, *index as f32, *value]),
                GameEvent::UiAction { node, action, value } => out.extend([12.0, *node as f32, *action as f32, *value]),
                GameEvent::ServiceAnswer { request, status, value, length } => out.extend([13.0, *request as f32, *status as f32, *value, *length as f32]),
                GameEvent::ServiceData { request, a, b, c } => out.extend([14.0, *request as f32, *a, *b, *c]),
                GameEvent::GamepadConnected { pad, standard } => out.extend([15.0, *pad as f32, *standard as f32]),
                GameEvent::GamepadDisconnected { pad } => out.extend([16.0, *pad as f32]),
                GameEvent::GamepadButton { button, pad, value, pressed } => out.extend([17.0, *button as f32, *pad as f32, *value, *pressed as f32]),
                GameEvent::GamepadAxis { axis, pad, value } => out.extend([18.0, *axis as f32, *pad as f32, *value]),
                GameEvent::Action { action, value, device } => out.extend([19.0, *action as f32, *value, *device as f32]),
                GameEvent::TextInput { character, composing } => out.extend([20.0, *character as f32, *composing as f32]),
                GameEvent::Composition { length, cursor } => out.extend([21.0, *length as f32, *cursor as f32]),
                GameEvent::MouseMotion { dx, dy } => out.extend([22.0, *dx, *dy]),
                GameEvent::Touch { id, x, y, phase } => out.extend([23.0, *id as f32, *x, *y, *phase as f32]),
                GameEvent::Pan { touches, dx, dy } => out.extend([24.0, *touches as f32, *dx, *dy]),
                GameEvent::Pinch { x, y, scale } => out.extend([25.0, *x, *y, *scale]),
                GameEvent::Player { index } => out.extend([26.0, *index as f32]),
                GameEvent::SessionStarted { restored } => out.extend([27.0, *restored as f32]),
                GameEvent::RendererReady => out.push(28.0),
                GameEvent::Game { event, a, b, c } => out.extend([100.0, *event as f32, *a, *b, *c]),
            }
        }
        out.push(state.len() as f32);
        out.extend_from_slice(state);
        out
    }

    mod canonical_sink {
        pub(super) trait CanonicalSink {
            fn push(&mut self, value: f32);
            fn extend<const N: usize>(&mut self, values: [f32; N]);
            fn extend_from_slice(&mut self, values: &[f32]);
        }

        impl CanonicalSink for Vec<f32> {
            #[inline]
            fn push(&mut self, value: f32) {
                Vec::push(self, value);
            }
            #[inline]
            fn extend<const N: usize>(&mut self, values: [f32; N]) {
                Vec::extend_from_slice(self, &values);
            }
            #[inline]
            fn extend_from_slice(&mut self, values: &[f32]) {
                Vec::extend_from_slice(self, values);
            }
        }

        pub(super) struct Fnv1aSink(pub(super) u64);

        impl Fnv1aSink {
            #[inline]
            fn write_lane(&mut self, lane: f32) {
                for byte in lane.to_bits().to_le_bytes() {
                    self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
                }
            }
        }

        impl CanonicalSink for Fnv1aSink {
            #[inline]
            fn push(&mut self, value: f32) {
                self.write_lane(value);
            }
            #[inline]
            fn extend<const N: usize>(&mut self, values: [f32; N]) {
                for value in values {
                    self.write_lane(value);
                }
            }
            #[inline]
            fn extend_from_slice(&mut self, values: &[f32]) {
                for &value in values {
                    self.write_lane(value);
                }
            }
        }
    }

    /// The walk of a render packet's values that pen's tests compare with the game's
    /// (`pen/protocol_canonical.sq`): the header's values and then the sections, each in the
    /// schema's order, as described in pen-codegen's canonical.rs. It takes the schema and
    /// the state, which the frame does not hold, where the packet has them.
    pub fn canonical_render(frame: &RenderFrame, schema: u32, state: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        walk_canonical_render(&mut out, frame, schema, state);
        out
    }

    /// Streaming FNV-1a hash of [`canonical_render`] without allocating a buffer.
    pub fn hash_canonical_render(frame: &RenderFrame, schema: u32, state: &[f32]) -> u64 {
        let mut out = canonical_sink::Fnv1aSink(0xcbf2_9ce4_8422_2325);
        walk_canonical_render(&mut out, frame, schema, state);
        out.0
    }

    fn walk_canonical_render<S: canonical_sink::CanonicalSink>(out: &mut S, frame: &RenderFrame, schema: u32, state: &[f32]) {
        out.extend([
            schema as f32, f32::from(frame.save_requested), f32::from(frame.load_requested),
            f32::from(frame.scene_save_requested), f32::from(frame.scene_load_requested),
            f32::from(frame.checkpoint_requested), f32::from(frame.undo_requested),
            f32::from(frame.redo_requested), f32::from(frame.exit_requested),
        ]);
        out.extend_from_slice(&frame.view);
        out.extend_from_slice(&frame.projection);
        out.extend([
            frame.clear.r, frame.clear.g, frame.clear.b, frame.clear.a, frame.ambient.x, frame.ambient.y,
            frame.ambient.z, frame.light_direction.x, frame.light_direction.y, frame.light_direction.z,
            frame.light_color.x, frame.light_color.y, frame.light_color.z, frame.aberration.x,
            frame.aberration.y, frame.quality.msaa as f32, f32::from(frame.quality.fxaa), frame.quality.scale,
            frame.quality.filtering as f32, f32::from(frame.quality.occlusion), f32::from(frame.quality.sky),
            f32::from(frame.quality.effects), f32::from(frame.quality.bloom), f32::from(frame.quality.taa),
            frame.quality.ssr as f32, frame.quality.volumetrics as f32, f32::from(frame.quality.ssgi),
            f32::from(frame.quality.oit), f32::from(frame.quality.physical_sky),
            f32::from(frame.quality.upscale), frame.quality.ray_tracing as f32, f32::from(frame.quality.ddgi),
            frame.sky.zenith.x, frame.sky.zenith.y, frame.sky.zenith.z, frame.sky.horizon.x, frame.sky.horizon.y,
            frame.sky.horizon.z, frame.sky.ground.x, frame.sky.ground.y, frame.sky.ground.z, frame.sky.sun.x,
            frame.sky.sun.y, frame.sky.sun.z, frame.sky.night, frame.sky.tint.x, frame.sky.tint.y,
            frame.sky.tint.z, frame.sky.sh0.x, frame.sky.sh0.y, frame.sky.sh0.z, frame.sky.sh1.x,
            frame.sky.sh1.y, frame.sky.sh1.z, frame.sky.sh2.x, frame.sky.sh2.y, frame.sky.sh2.z, frame.sky.sh3.x,
            frame.sky.sh3.y, frame.sky.sh3.z, frame.sky.sh4.x, frame.sky.sh4.y, frame.sky.sh4.z, frame.sky.sh5.x,
            frame.sky.sh5.y, frame.sky.sh5.z, frame.sky.sh6.x, frame.sky.sh6.y, frame.sky.sh6.z, frame.sky.sh7.x,
            frame.sky.sh7.y, frame.sky.sh7.z, frame.sky.sh8.x, frame.sky.sh8.y, frame.sky.sh8.z,
            frame.probe_volume.revision as f32, frame.probe_volume.nx as f32, frame.probe_volume.ny as f32,
            frame.probe_volume.nz as f32, frame.probe_volume.spacing, frame.probe_volume.origin.x,
            frame.probe_volume.origin.y, frame.probe_volume.origin.z, frame.fog.radius, frame.fog.density,
            frame.fog.height, frame.fog.edge, frame.fog.distance, frame.fog.albedo.x, frame.fog.albedo.y,
            frame.fog.albedo.z, frame.fog.anisotropy, frame.fog.sun, frame.fog.ambient, frame.fog.sky,
            frame.fog.noise, frame.fog.noise_scale, frame.fog.drift.x, frame.fog.drift.y, frame.fog.drift.z,
        ]);
        out.extend_from_slice(&frame.shadow.view);
        out.extend([
            frame.shadow.texels as f32, frame.shadow.filter as f32, frame.shadow.softness,
            frame.shadow.contact as f32, frame.shadow.contact_length, frame.shadow.local as f32,
            frame.shadow.local_texels as f32, frame.blur, frame.clouds.cover, frame.clouds.drift.x,
            frame.clouds.drift.z, f32::from(frame.hide_cursor), f32::from(frame.loading),
            f32::from(frame.show_captions), f32::from(frame.retained_enabled), f32::from(frame.retained_reset),
            frame.retained_after as f32, f32::from(frame.exposure.auto), frame.exposure.compensation,
            frame.exposure.min, frame.exposure.max, frame.exposure.brighten, frame.exposure.darken,
            frame.exposure.low, frame.exposure.high, frame.exposure.centre, frame.exposure.range_low,
            frame.exposure.range_high, frame.exposure.target, frame.exposure.dt, f32::from(frame.exposure.cut),
            frame.grade.temperature, frame.grade.tint, frame.grade.exposure, frame.grade.contrast,
            frame.grade.pivot, frame.grade.saturation, frame.grade.vibrance, frame.grade.lift.x,
            frame.grade.lift.y, frame.grade.lift.z, frame.grade.gamma.x, frame.grade.gamma.y,
            frame.grade.gamma.z, frame.grade.gain.x, frame.grade.gain.y, frame.grade.gain.z,
            frame.grade.lut_a as f32, frame.grade.lut_b as f32, frame.grade.lut_blend, frame.grade.lut_weight,
            frame.grade.vignette, frame.grade.vignette_radius, frame.grade.tonemap as f32, frame.grade.sharpness,
            frame.grade.grain, frame.grade.flare, frame.grade.outline_strength, frame.reflection.budget as f32,
            frame.motion_blur.strength, frame.motion_blur.max_radius, frame.motion_blur.samples as f32,
            frame.dof.focus_distance, frame.dof.focus_range, frame.dof.aperture, frame.dof.max_blur,
            frame.reverb.decay, frame.reverb.damping, state.len() as f32,
        ]);
        out.extend_from_slice(state);
        out.push(frame.draws.len() as f32);
        for draw in &frame.draws {
            canonical_draw(out, draw);
        }
        out.push(frame.solids.len() as f32);
        for solid in &frame.solids {
            canonical_solid(out, solid);
        }
        out.push(frame.decals.len() as f32);
        for decal in &frame.decals {
            canonical_decal(out, decal);
        }
        out.push(frame.sounds.len() as f32);
        for sound in &frame.sounds {
            canonical_sound(out, sound);
        }
        out.push(frame.sprites.len() as f32);
        for sprite in &frame.sprites {
            canonical_sprite(out, sprite);
        }
        out.push(frame.insets.len() as f32);
        for inset in &frame.insets {
            canonical_inset(out, inset);
        }
        match &frame.preferences {
            Some(values) => {
                out.extend([1.0, values.len() as f32]);
                out.extend_from_slice(values);
            }
            None => out.push(0.0),
        }
        out.push(frame.fields.len() as f32);
        for field_patch in &frame.fields {
            canonical_field_patch(out, field_patch);
        }
        out.push(frame.ui.len() as f32);
        for ui_node in &frame.ui {
            canonical_ui_node(out, ui_node);
        }
        out.push(frame.services.len() as f32);
        for service_request in &frame.services {
            canonical_service_request(out, service_request);
        }
        out.push(frame.buses.len() as f32);
        for bus in &frame.buses {
            canonical_bus(out, bus);
        }
        out.push(frame.voices.len() as f32);
        for voice in &frame.voices {
            canonical_voice(out, voice);
        }
        out.push(frame.poses.len() as f32);
        for pose in &frame.poses {
            canonical_pose(out, pose);
        }
        out.push(frame.emitters.len() as f32);
        for emitter in &frame.emitters {
            canonical_emitter(out, emitter);
        }
        out.push(frame.lights.len() as f32);
        for light in &frame.lights {
            canonical_light(out, light);
        }
        out.push(frame.cascades.len() as f32);
        for cascade in &frame.cascades {
            canonical_cascade(out, cascade);
        }
        out.push(frame.probes.len() as f32);
        for probe in &frame.probes {
            canonical_probe(out, probe);
        }
        out.push(frame.reflection_probes.len() as f32);
        for reflection_probe in &frame.reflection_probes {
            canonical_reflection_probe(out, reflection_probe);
        }
        out.push(frame.fog_volumes.len() as f32);
        for fog_volume in &frame.fog_volumes {
            canonical_fog_volume(out, fog_volume);
        }
        out.push(frame.listener.len() as f32);
        for listener in &frame.listener {
            canonical_listener(out, listener);
        }
        out.push(frame.panners.len() as f32);
        for panner in &frame.panners {
            canonical_panner(out, panner);
        }
        out.push(frame.instances.len() as f32);
        for instance in &frame.instances {
            canonical_instance(out, instance);
        }
        out.push(frame.instance_removals.len() as f32);
        for instance_removal in &frame.instance_removals {
            canonical_instance_removal(out, instance_removal);
        }
        out.push(frame.bone_matrices.len() as f32);
        for bone_matrix in &frame.bone_matrices {
            canonical_bone_matrix(out, bone_matrix);
        }
        out.push(frame.morph_weights.len() as f32);
        for morph_weight in &frame.morph_weights {
            canonical_morph_weight(out, morph_weight);
        }
        out.push(frame.motions.len() as f32);
        for motion in &frame.motions {
            canonical_motion(out, motion);
        }
        out.push(frame.ribbons.len() as f32);
        for ribbon_point in &frame.ribbons {
            canonical_ribbon_point(out, ribbon_point);
        }
        out.push(frame.haptics.len() as f32);
        for haptic in &frame.haptics {
            canonical_haptic(out, haptic);
        }
        out.push(frame.reverb_zones.len() as f32);
        for reverb_zone in &frame.reverb_zones {
            canonical_reverb_zone(out, reverb_zone);
        }
        out.push(frame.synths.len() as f32);
        for synth_voice in &frame.synths {
            canonical_synth_voice(out, synth_voice);
        }
        out.push(frame.area_lights.len() as f32);
        for area_light_spec in &frame.area_lights {
            canonical_area_light_spec(out, area_light_spec);
        }
        out.push(frame.pbr_decals.len() as f32);
        for pbr_decal_spec in &frame.pbr_decals {
            canonical_pbr_decal_spec(out, pbr_decal_spec);
        }
        out.push(frame.camera_targets.len() as f32);
        for camera_target_spec in &frame.camera_targets {
            canonical_camera_target_spec(out, camera_target_spec);
        }
        out.push(frame.video_streams.len() as f32);
        for video_stream_spec in &frame.video_streams {
            canonical_video_stream_spec(out, video_stream_spec);
        }
    }

    fn canonical_draw<S: canonical_sink::CanonicalSink>(out: &mut S, draw: &Draw) {
        out.extend([draw.mesh as f32, draw.material as f32]);
        out.extend_from_slice(&draw.transform);
        match &draw.parameters {
            Some(values) => {
                out.push(1.0);
                out.extend_from_slice(values);
            }
            None => out.push(0.0),
        }
        match &draw.extra {
            Some(values) => {
                out.extend([1.0, values.len() as f32]);
                out.extend_from_slice(values);
            }
            None => out.push(0.0),
        }
    }

    fn canonical_solid_face<S: canonical_sink::CanonicalSink>(out: &mut S, solid_face: &SolidFace) {
        out.extend([
            solid_face.normal.x, solid_face.normal.y, solid_face.normal.z, solid_face.distance,
            solid_face.material as f32,
        ]);
    }

    fn canonical_solid<S: canonical_sink::CanonicalSink>(out: &mut S, solid: &Solid) {
        out.push(solid.faces.len() as f32);
        for solid_face in &solid.faces {
            canonical_solid_face(out, solid_face);
        }
    }

    fn canonical_decal<S: canonical_sink::CanonicalSink>(out: &mut S, decal: &Decal) {
        out.push(decal.material as f32);
        out.extend_from_slice(&decal.transform);
        out.push(decal.opacity);
    }

    fn canonical_sound<S: canonical_sink::CanonicalSink>(out: &mut S, sound: &Sound) {
        out.extend([
            sound.clip as f32, sound.bus as f32, sound.gain, sound.pan, sound.pitch, sound.lowpass, sound.reverb,
        ]);
    }

    fn canonical_sprite<S: canonical_sink::CanonicalSink>(out: &mut S, sprite: &Sprite) {
        out.extend([
            sprite.x, sprite.y, sprite.width, sprite.height, sprite.u0, sprite.v0, sprite.u1, sprite.v1,
        ]);
        out.extend_from_slice(&sprite.color);
        out.push(sprite.material as f32);
        out.extend_from_slice(&sprite.parameters);
    }

    fn canonical_inset<S: canonical_sink::CanonicalSink>(out: &mut S, inset: &Inset) {
        out.extend([inset.x, inset.y, inset.width, inset.height]);
        out.extend_from_slice(&inset.view);
        out.extend_from_slice(&inset.projection);
        out.extend([
            inset.ambient.x, inset.ambient.y, inset.ambient.z, inset.light_direction.x, inset.light_direction.y,
            inset.light_direction.z, inset.light_color.x, inset.light_color.y, inset.light_color.z,
            inset.draws.len() as f32,
        ]);
        for draw in &inset.draws {
            canonical_draw(out, draw);
        }
        out.extend([inset.pixel_size, inset.after_sprites as f32]);
    }

    fn canonical_field_patch<S: canonical_sink::CanonicalSink>(out: &mut S, field_patch: &FieldPatch) {
        out.extend([
            field_patch.material as f32, field_patch.width as f32, field_patch.height as f32,
            field_patch.x as f32, field_patch.z as f32, field_patch.columns as f32, field_patch.rows as f32,
            field_patch.values.len() as f32,
        ]);
        out.extend_from_slice(&field_patch.values);
    }

    fn canonical_ui_node<S: canonical_sink::CanonicalSink>(out: &mut S, ui_node: &UiNode) {
        out.extend([
            ui_node.id as f32, ui_node.parent as f32, ui_node.role as f32, ui_node.x, ui_node.y, ui_node.width,
            ui_node.height, f32::from(ui_node.focusable), f32::from(ui_node.focused),
            f32::from(ui_node.disabled), f32::from(ui_node.hidden), f32::from(ui_node.modal),
            ui_node.checked as f32, ui_node.expanded as f32, ui_node.selected as f32, ui_node.live as f32,
            ui_node.value, ui_node.minimum, ui_node.maximum, ui_node.step, ui_node.actions as f32,
            ui_node.text_length as f32, ui_node.name_length as f32, ui_node.description_length as f32,
            ui_node.text.len() as f32,
        ]);
        out.extend_from_slice(&ui_node.text);
    }

    fn canonical_service_request<S: canonical_sink::CanonicalSink>(out: &mut S, service_request: &ServiceRequest) {
        out.extend([
            service_request.request as f32, service_request.service as f32, service_request.operation as f32,
            service_request.a, service_request.b, service_request.c, service_request.d,
            service_request.value_count as f32, service_request.values.len() as f32,
        ]);
        out.extend_from_slice(&service_request.values);
    }

    fn canonical_bus<S: canonical_sink::CanonicalSink>(out: &mut S, bus: &Bus) {
        out.extend([
            bus.gain, bus.lowpass, bus.reverb, bus.comp_threshold, bus.comp_ratio, bus.duck_bus as f32,
            bus.duck_amount, bus.eq_low_db, bus.eq_mid_db, bus.eq_high_db,
        ]);
    }

    fn canonical_voice<S: canonical_sink::CanonicalSink>(out: &mut S, voice: &Voice) {
        out.extend([
            voice.id as f32, voice.clip as f32, voice.bus as f32, voice.gain, voice.pan, voice.pitch,
            voice.lowpass, voice.reverb, f32::from(voice.looping), voice.start, voice.fade_in, voice.fade_out,
        ]);
    }

    fn canonical_pose<S: canonical_sink::CanonicalSink>(out: &mut S, pose: &Pose) {
        out.extend([pose.draw as f32, pose.clip as f32, pose.time, pose.weight, pose.inset as f32]);
    }

    fn canonical_emitter<S: canonical_sink::CanonicalSink>(out: &mut S, emitter: &Emitter) {
        out.push(emitter.material as f32);
        out.extend_from_slice(&emitter.transform);
        out.extend([
            emitter.seed as f32, emitter.time, emitter.duration, emitter.rate, emitter.lifetime, emitter.radius,
            emitter.velocity.x, emitter.velocity.y, emitter.velocity.z, emitter.spread, emitter.gravity.x,
            emitter.gravity.y, emitter.gravity.z, emitter.drag, emitter.size_start, emitter.size_end,
            emitter.color_start.r, emitter.color_start.g, emitter.color_start.b, emitter.color_start.a,
            emitter.color_end.r, emitter.color_end.g, emitter.color_end.b, emitter.color_end.a,
            f32::from(emitter.additive), emitter.soft_depth, emitter.lit, emitter.flipbook_cols as f32,
            emitter.flipbook_rows as f32, emitter.flipbook_fps,
        ]);
    }

    fn canonical_light<S: canonical_sink::CanonicalSink>(out: &mut S, light: &Light) {
        out.extend([
            light.position.x, light.position.y, light.position.z, light.radius, light.color.x, light.color.y,
            light.color.z, light.direction.x, light.direction.y, light.direction.z, light.inner, light.outer,
            f32::from(light.shadow), light.id as f32,
        ]);
    }

    fn canonical_cascade<S: canonical_sink::CanonicalSink>(out: &mut S, cascade: &Cascade) {
        out.extend_from_slice(&cascade.projection);
        out.extend([cascade.texel, cascade.far]);
    }

    fn canonical_probe<S: canonical_sink::CanonicalSink>(out: &mut S, probe: &Probe) {
        out.extend([
            probe.sh0.x, probe.sh0.y, probe.sh0.z, probe.sh1.x, probe.sh1.y, probe.sh1.z, probe.sh2.x,
            probe.sh2.y, probe.sh2.z, probe.sh3.x, probe.sh3.y, probe.sh3.z, probe.sh4.x, probe.sh4.y,
            probe.sh4.z, probe.sh5.x, probe.sh5.y, probe.sh5.z, probe.sh6.x, probe.sh6.y, probe.sh6.z,
            probe.sh7.x, probe.sh7.y, probe.sh7.z, probe.sh8.x, probe.sh8.y, probe.sh8.z, probe.valid,
        ]);
    }

    fn canonical_reflection_probe<S: canonical_sink::CanonicalSink>(out: &mut S, reflection_probe: &ReflectionProbe) {
        out.extend([
            reflection_probe.id as f32, reflection_probe.revision as f32, reflection_probe.priority as f32,
            f32::from(reflection_probe.capture), reflection_probe.position.x, reflection_probe.position.y,
            reflection_probe.position.z, reflection_probe.center.x, reflection_probe.center.y,
            reflection_probe.center.z, reflection_probe.extent.x, reflection_probe.extent.y,
            reflection_probe.extent.z, reflection_probe.blend, reflection_probe.parallax_center.x,
            reflection_probe.parallax_center.y, reflection_probe.parallax_center.z,
            reflection_probe.parallax_extent.x, reflection_probe.parallax_extent.y,
            reflection_probe.parallax_extent.z, reflection_probe.far,
        ]);
    }

    fn canonical_fog_volume<S: canonical_sink::CanonicalSink>(out: &mut S, fog_volume: &FogVolume) {
        out.extend([
            fog_volume.shape as f32, fog_volume.center.x, fog_volume.center.y, fog_volume.center.z,
            fog_volume.extent.x, fog_volume.extent.y, fog_volume.extent.z, fog_volume.density,
            fog_volume.albedo.x, fog_volume.albedo.y, fog_volume.albedo.z, fog_volume.emission.x,
            fog_volume.emission.y, fog_volume.emission.z, fog_volume.falloff,
        ]);
    }

    fn canonical_listener<S: canonical_sink::CanonicalSink>(out: &mut S, listener: &Listener) {
        out.extend([
            listener.position.x, listener.position.y, listener.position.z, listener.forward.x,
            listener.forward.y, listener.forward.z, listener.up.x, listener.up.y, listener.up.z,
            listener.velocity.x, listener.velocity.y, listener.velocity.z,
        ]);
    }

    fn canonical_panner<S: canonical_sink::CanonicalSink>(out: &mut S, panner: &Panner) {
        out.extend([
            panner.voice as f32, panner.sound as f32, panner.position.x, panner.position.y, panner.position.z,
            panner.orientation.x, panner.orientation.y, panner.orientation.z, panner.distance_model as f32,
            panner.ref_distance, panner.max_distance, panner.rolloff, panner.cone_inner, panner.cone_outer,
            panner.cone_outer_gain, f32::from(panner.hrtf), panner.velocity.x, panner.velocity.y,
            panner.velocity.z, panner.doppler, panner.occlusion, panner.zone as f32,
        ]);
    }

    fn canonical_instance<S: canonical_sink::CanonicalSink>(out: &mut S, instance: &Instance) {
        out.extend([
            instance.id_low as f32, instance.id_high as f32, instance.order as f32, instance.mesh as f32,
            instance.material as f32,
        ]);
        out.extend_from_slice(&instance.transform);
        match &instance.parameters {
            Some(values) => {
                out.push(1.0);
                out.extend_from_slice(values);
            }
            None => out.push(0.0),
        }
    }

    fn canonical_instance_removal<S: canonical_sink::CanonicalSink>(out: &mut S, instance_removal: &InstanceRemoval) {
        out.extend([instance_removal.id_low as f32, instance_removal.id_high as f32]);
    }

    fn canonical_bone_matrix<S: canonical_sink::CanonicalSink>(out: &mut S, bone_matrix: &BoneMatrix) {
        out.extend([bone_matrix.inset as f32, bone_matrix.draw as f32, bone_matrix.joint as f32]);
        out.extend_from_slice(&bone_matrix.matrix);
    }

    fn canonical_morph_weight<S: canonical_sink::CanonicalSink>(out: &mut S, morph_weight: &MorphWeight) {
        out.extend([
            morph_weight.inset as f32, morph_weight.draw as f32, morph_weight.target as f32, morph_weight.weight,
        ]);
    }

    fn canonical_motion<S: canonical_sink::CanonicalSink>(out: &mut S, motion: &Motion) {
        out.push(motion.draw as f32);
        out.extend_from_slice(&motion.previous);
    }

    fn canonical_ribbon_point<S: canonical_sink::CanonicalSink>(out: &mut S, ribbon_point: &RibbonPoint) {
        out.extend([
            ribbon_point.ribbon as f32, ribbon_point.material as f32, f32::from(ribbon_point.additive),
            ribbon_point.position.x, ribbon_point.position.y, ribbon_point.position.z, ribbon_point.width,
            ribbon_point.color.r, ribbon_point.color.g, ribbon_point.color.b, ribbon_point.color.a,
            ribbon_point.soft_depth,
        ]);
    }

    fn canonical_haptic<S: canonical_sink::CanonicalSink>(out: &mut S, haptic: &Haptic) {
        out.extend([haptic.pad as f32, haptic.low, haptic.high, haptic.duration]);
    }

    fn canonical_reverb_zone<S: canonical_sink::CanonicalSink>(out: &mut S, reverb_zone: &ReverbZone) {
        out.extend([
            reverb_zone.id as f32, reverb_zone.position.x, reverb_zone.position.y, reverb_zone.position.z,
            reverb_zone.half_extents.x, reverb_zone.half_extents.y, reverb_zone.half_extents.z,
            reverb_zone.blend, reverb_zone.decay, reverb_zone.damping, reverb_zone.wet,
            reverb_zone.portal_attenuation,
        ]);
    }

    fn canonical_synth_voice<S: canonical_sink::CanonicalSink>(out: &mut S, synth_voice: &SynthVoice) {
        out.extend([
            synth_voice.id as f32, synth_voice.mode as f32, synth_voice.bus as f32, synth_voice.frequency,
            synth_voice.gain, synth_voice.pan, synth_voice.detune, synth_voice.attack, synth_voice.decay,
            synth_voice.sustain, synth_voice.release, synth_voice.cutoff, synth_voice.resonance,
            synth_voice.grain_size, synth_voice.grain_pitch, synth_voice.reverb,
        ]);
    }

    fn canonical_area_light_spec<S: canonical_sink::CanonicalSink>(out: &mut S, area_light_spec: &AreaLightSpec) {
        out.extend([
            area_light_spec.kind as f32, area_light_spec.position.x, area_light_spec.position.y,
            area_light_spec.position.z, area_light_spec.right.x, area_light_spec.right.y,
            area_light_spec.right.z, area_light_spec.up.x, area_light_spec.up.y, area_light_spec.up.z,
            area_light_spec.color.x, area_light_spec.color.y, area_light_spec.color.z, area_light_spec.intensity,
            area_light_spec.width, area_light_spec.height, area_light_spec.range,
            area_light_spec.cookie_material as f32, f32::from(area_light_spec.two_sided),
        ]);
    }

    fn canonical_pbr_decal_spec<S: canonical_sink::CanonicalSink>(out: &mut S, pbr_decal_spec: &PbrDecalSpec) {
        out.push(pbr_decal_spec.material as f32);
        out.extend_from_slice(&pbr_decal_spec.transform);
        out.extend([
            pbr_decal_spec.albedo.r, pbr_decal_spec.albedo.g, pbr_decal_spec.albedo.b, pbr_decal_spec.albedo.a,
            pbr_decal_spec.normal_strength, pbr_decal_spec.roughness, pbr_decal_spec.metallic,
            pbr_decal_spec.emissive_strength, pbr_decal_spec.fade,
        ]);
    }

    fn canonical_camera_target_spec<S: canonical_sink::CanonicalSink>(out: &mut S, camera_target_spec: &CameraTargetSpec) {
        out.extend([
            camera_target_spec.material as f32, camera_target_spec.width as f32,
            camera_target_spec.height as f32, camera_target_spec.viewport_x, camera_target_spec.viewport_y,
            camera_target_spec.viewport_w, camera_target_spec.viewport_h,
            f32::from(camera_target_spec.split_screen),
        ]);
        out.extend_from_slice(&camera_target_spec.view);
        out.extend_from_slice(&camera_target_spec.projection);
    }

    fn canonical_video_stream_spec<S: canonical_sink::CanonicalSink>(out: &mut S, video_stream_spec: &VideoStreamSpec) {
        out.extend([
            video_stream_spec.material as f32, video_stream_spec.time, f32::from(video_stream_spec.playing),
            f32::from(video_stream_spec.looping), video_stream_spec.speed,
        ]);
    }
}
