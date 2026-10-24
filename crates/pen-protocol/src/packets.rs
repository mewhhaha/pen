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
    /// test-blot --session.
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
    /// sounds that play (CAPTIONS_OP_*) and the player's consent to crash and performance
    /// reports (REPORTS_OP_*). A host without one answers unavailable.
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
        /// The names of the codes, by code.
        pub const NAMES: [&str; 15] = ["steam", "clipboard", "url", "platform", "pathfinding", "devtools", "physics", "navmesh", "spatial", "noise", "input", "preferences", "locale", "captions", "reports"];
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
        /// generated pen/extensions.blot decodes them; pen's own bindings leave them out.
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
    }
    /// A listener: where it is, the way it faces and its up, in world units.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Listener {
        pub position: Vector,
        /// The way it faces.
        pub forward: Vector,
        /// Its up, which with `forward` tells left from right.
        pub up: Vector,
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
    }
    /// How the game wants this frame drawn: samples per pixel, FXAA, render scale, texture
    /// filtering, ambient occlusion, a sky, screen effects, bloom and temporal antialiasing.
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
        /// Temporal antialiasing: each frame's samples move within their pixels and blend with
        /// the frames before, followed through the camera's motion.
        pub taa: bool,
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
    /// One of the game's own sections: its tag, from its declaration, and its lanes.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct Extension {
        pub tag: u32,
        pub values: Vec<f32>,
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
        /// filtering, ambient occlusion, a sky, screen effects, bloom and temporal
        /// antialiasing.
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
        /// The frame sprites are an application loading screen: draw them above the startup
        /// veil even while scene pipelines compile. Scene assets continue loading underneath.
        pub loading: bool,
        /// Show sound captions when the player has enabled them in the game's accessibility
        /// settings. Off by default; the game owns and saves the choice.
        pub show_captions: bool,
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
        /// Where the panners are heard from, as WebAudio's AudioListener: none, the camera
        /// (`view`'s eye, facing where it looks, its up the view's), or one listener.
        pub listener: Vec<Listener>,
        /// Voices and sounds heard from somewhere in the world, as WebAudio's PannerNode
        /// (equal-power panning) places them relative to the listener; the rest play as they
        /// are.
        pub panners: Vec<Panner>,
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
    /// Preference events from this index on carry the engine's preferences
    /// (ENGINE_PREFERENCE_*); below it, the game's own.
    pub const ENGINE_PREFERENCES: usize = 64;
    /// The most panners one frame may declare: one for each voice and sound it may have.
    pub const MAX_PANNERS: usize = 512;
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
    /// sky.sun3, sky.night, fog.radius, fog.density, fog.height, shadow.view16,
    /// shadow.projection16, shadow.texel, shadow.texels, blur, quality.bloom, clouds.cover,
    /// clouds.drift2, hide_cursor, field_count, field_lanes, ui_count, ui_lanes, service_count,
    /// service_lanes, bus_count, voice_count, pose_count, emitter_count, quality.taa,
    /// light_count, listener_count, panner_count, loading, show_captions.
    pub const PACKET_HEADER_FLOATS: usize = 146;
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
    /// Lanes: gain, lowpass, reverb.
    pub const BUS_FLOATS: usize = 3;
    /// Lanes: id, clip, bus, gain, pan, pitch, lowpass, reverb, looping, start, fade_in,
    /// fade_out.
    pub const VOICE_FLOATS: usize = 12;
    /// Lanes: draw, clip, time, weight.
    pub const POSE_FLOATS: usize = 4;
    /// Lanes: material, transform16, seed, time, duration, rate, lifetime, radius, velocity3,
    /// spread, gravity3, drag, size_start, size_end, color_start4, color_end4, additive.
    pub const EMITTER_FLOATS: usize = 42;
    /// Lanes: position3, radius, color3, direction3, inner, outer.
    pub const LIGHT_FLOATS: usize = 12;
    /// Lanes: position3, forward3, up3.
    pub const LISTENER_FLOATS: usize = 9;
    /// Lanes: voice, sound, position3, orientation3, distance_model, ref_distance,
    /// max_distance, rolloff, cone_inner, cone_outer, cone_outer_gain.
    pub const PANNER_FLOATS: usize = 15;
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
            pub const POSE_COUNT: usize = 138;
            pub const EMITTER_COUNT: usize = 139;
            pub const QUALITY_TAA: usize = 140;
            pub const LIGHT_COUNT: usize = 141;
            pub const LISTENER_COUNT: usize = 142;
            pub const PANNER_COUNT: usize = 143;
            pub const LOADING: usize = 144;
            pub const SHOW_CAPTIONS: usize = 145;
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
        }
        pub mod light {
            pub const POSITION: usize = 0;
            pub const RADIUS: usize = 3;
            pub const COLOR: usize = 4;
            pub const DIRECTION: usize = 7;
            pub const INNER: usize = 10;
            pub const OUTER: usize = 11;
        }
        pub mod listener {
            pub const POSITION: usize = 0;
            pub const FORWARD: usize = 3;
            pub const UP: usize = 6;
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
            let after_sprites = self.count(lane + 47, MAX_SPRITES)?;
            *inset = Inset { x, y, width, height, view, projection, ambient, light_direction, light_color, draws, pixel_size, after_sprites: after_sprites as u32 };
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

        fn decode_pose(&mut self, lane: usize, pose: &mut Pose) -> Result<()> {
            let draw = self.count(lane, MAX_DRAWS)?;
            let clip = self.count(lane + 1, 65535)?;
            let time = self.packet[lane + 2];
            let weight = self.packet[lane + 3];
            *pose = Pose { draw: draw as u32, clip: clip as u32, time, weight };
            Ok(())
        }

        fn decode_emitter(&mut self, lane: usize, emitter: &mut Emitter) -> Result<()> {
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
            *emitter = Emitter { material, transform, seed: seed as u32, time, duration, rate, lifetime, radius, velocity, spread, gravity, drag, size_start, size_end, color_start, color_end, additive };
            Ok(())
        }

        fn decode_light(&mut self, lane: usize, light: &mut Light) -> Result<()> {
            let position = Vector { x: self.packet[lane], y: self.packet[lane + 1], z: self.packet[lane + 2] };
            let radius = self.packet[lane + 3];
            let color = Vector { x: self.packet[lane + 4], y: self.packet[lane + 5], z: self.packet[lane + 6] };
            let direction = Vector { x: self.packet[lane + 7], y: self.packet[lane + 8], z: self.packet[lane + 9] };
            let inner = self.packet[lane + 10];
            let outer = self.packet[lane + 11];
            *light = Light { position, radius, color, direction, inner, outer };
            Ok(())
        }

        fn decode_listener(&mut self, lane: usize, listener: &mut Listener) -> Result<()> {
            let position = Vector { x: self.packet[lane], y: self.packet[lane + 1], z: self.packet[lane + 2] };
            let forward = Vector { x: self.packet[lane + 3], y: self.packet[lane + 4], z: self.packet[lane + 5] };
            let up = Vector { x: self.packet[lane + 6], y: self.packet[lane + 7], z: self.packet[lane + 8] };
            *listener = Listener { position, forward, up };
            Ok(())
        }

        fn decode_panner(&mut self, lane: usize, panner: &mut Panner) -> Result<()> {
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
            *panner = Panner { voice: voice as u32, sound: sound as u32, position, orientation, distance_model: distance_model as u32, ref_distance, max_distance, rolloff, cone_inner, cone_outer, cone_outer_gain };
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
        let fields_count = reader.count(130, MAX_FIELD_PATCHES)?;
        let fields_lanes = reader.count(131, 16805888)?;
        let ui_count = reader.count(132, MAX_UI_NODES)?;
        let ui_lanes = reader.count(133, 4292608)?;
        let services_count = reader.count(134, MAX_SERVICE_REQUESTS)?;
        let services_lanes = reader.count(135, 4196352)?;
        let buses_count = reader.count(136, MAX_BUSES)?;
        let voices_count = reader.count(137, MAX_VOICES)?;
        let poses_count = reader.count(138, MAX_POSES)?;
        let emitters_count = reader.count(139, MAX_EMITTERS)?;
        let lights_count = reader.count(141, MAX_LIGHTS)?;
        let listener_count = reader.count(142, 1)?;
        let panners_count = reader.count(143, MAX_PANNERS)?;
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
        let listener_start = lights_end;
        let listener_end = listener_start + listener_count * LISTENER_FLOATS;
        let panners_start = listener_end;
        let panners_end = panners_start + panners_count * PANNER_FLOATS;
        let extras_start = panners_end;
        let truncated = "game returned a truncated, oversized or nonfinite frame";
        ensure!(packet.len() >= panners_end && packet.iter().all(|value| value.is_finite()), truncated);
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
        extensions.clear();
        let mut extensions_end = extras_end;
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
                    ensure!(!provenance_seen, "duplicate development provenance");
                    provenance_seen = true;
                    crate::diagnostic::read_provenance(&packet[first..first + length], extras_end).map_err(|_| anyhow::anyhow!("invalid development provenance"))?;
                } else {
                    extensions.push(Extension { tag: tag as u32, values: packet[first..first + length].to_vec() });
                }
                extensions_end = first + length;
            }
        }
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
        let quality_taa = reader.flag(140)?;
        let loading = reader.flag(144)?;
        let show_captions = reader.flag(145)?;
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
            let service = reader.count(service_lane + 1, 14)?;
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
            quality: Quality { msaa: quality_msaa, fxaa: quality_fxaa, scale: quality_scale, filtering: quality_filtering as u32, occlusion: quality_occlusion, sky: quality_sky, effects: quality_effects, bloom: quality_bloom, taa: quality_taa },
            sky: Sky { zenith: sky_zenith, horizon: sky_horizon, ground: sky_ground, sun: sky_sun, night: sky_night },
            fog: Fog { radius: fog_radius, density: fog_density, height: fog_height },
            shadow: Shadow { view: shadow_view, projection: shadow_projection, texel: shadow_texel, texels: shadow_texels as u32 },
            blur,
            clouds: Clouds { cover: clouds_cover, drift: clouds_drift },
            hide_cursor,
            loading,
            show_captions,
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
            listener,
            panners,
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

    use crate::diagnostic::{Actual, Check, FrameDiagnostic, Inspector, Location};
    #[allow(unused_variables)]
    fn inspect_draw(d: &Inspector<'_>, catalog: &Catalog, base: usize, path: &Location, extras_cursor: &mut usize) -> Check {
        d.reference(&path.field("mesh"), base, catalog.meshes.len(), true)?;
        d.reference(&path.field("material"), base + 1, catalog.materials.len(), true)?;
        d.values(&path.field("transform"), base + 2, 16, &[])?;
        let count = d.count(&path.field("parameters").field("count"), base + 18, 4)?;
        if count != 0 && count != 4 { return Err(path.field("parameters").field("count").error(base + 18, "0 or 4", Actual::Value(count as f32))); }
        d.values(&path.field("parameters"), base + 18 + 1, 4, &[])?;
        let count = d.count(&path.field("extra"), base + 23, MAX_DRAW_EXTRA)?;
        let end = d.end(&path.field("extra"), *extras_cursor, count, 4)?;
        d.values(&path.field("extra"), *extras_cursor, end - *extras_cursor, &[])?;
        *extras_cursor = end;
        Ok(())
    }
    #[allow(unused_variables)]
    fn inspect_solid_face(d: &Inspector<'_>, catalog: &Catalog, base: usize, path: &Location, extras_cursor: &mut usize) -> Check {
        d.values(&path.field("normal"), base, 3, &["x","y","z"])?;
        d.finite(&path.field("distance"), base + 3)?;
        d.reference(&path.field("material"), base + 4, catalog.materials.len(), true)?;
        Ok(())
    }
    #[allow(unused_variables)]
    pub(crate) fn inspect_frame(packet: &[f32], schema: u32, catalog: &Catalog) -> Check {
        let d = &Inspector { packet };
        let header = Location::section("header");
        d.equal(&header.field("magic"), 0, PACKET_MAGIC)?;
        d.equal(&header.field("schema"), 1, schema as f32)?;
        let state_lanes = d.count(&header.field("state_count"), 2, MAX_STATE_FLOATS)?;
        let draws_count = d.count(&header.field("draw_count"), 3, MAX_DRAWS)?;
        let solids_count = d.count(&header.field("solid_count"), 4, MAX_SOLIDS)?;
        let solids_lanes = d.count(&header.field("solid_lanes"), 5, 32100000)?;
        let decals_count = d.count(&header.field("decal_count"), 6, MAX_DECALS)?;
        let sounds_count = d.count(&header.field("sound_count"), 7, MAX_SOUNDS)?;
        let sprites_count = d.count(&header.field("sprite_count"), 8, MAX_SPRITES)?;
        let insets_count = d.count(&header.field("inset_count"), 9, MAX_INSETS)?;
        let inset_draws_count = d.count(&header.field("inset_draw_count"), 10, MAX_DRAWS)?;
        let preferences_lanes = d.count(&header.field("preference_count"), 11, 256)?;
        let fields_count = d.count(&header.field("field_count"), 130, MAX_FIELD_PATCHES)?;
        let fields_lanes = d.count(&header.field("field_lanes"), 131, 16805888)?;
        let ui_count = d.count(&header.field("ui_count"), 132, MAX_UI_NODES)?;
        let ui_lanes = d.count(&header.field("ui_lanes"), 133, 4292608)?;
        let services_count = d.count(&header.field("service_count"), 134, MAX_SERVICE_REQUESTS)?;
        let services_lanes = d.count(&header.field("service_lanes"), 135, 4196352)?;
        let buses_count = d.count(&header.field("bus_count"), 136, MAX_BUSES)?;
        let voices_count = d.count(&header.field("voice_count"), 137, MAX_VOICES)?;
        let poses_count = d.count(&header.field("pose_count"), 138, MAX_POSES)?;
        let emitters_count = d.count(&header.field("emitter_count"), 139, MAX_EMITTERS)?;
        let lights_count = d.count(&header.field("light_count"), 141, MAX_LIGHTS)?;
        let listener_count = d.count(&header.field("listener_count"), 142, 1)?;
        let panners_count = d.count(&header.field("panner_count"), 143, MAX_PANNERS)?;
        let state_start = PACKET_HEADER_FLOATS;
        let state_end = d.end(&Location::section("state"), state_start, state_lanes, 1)?;
        let draws_start = state_end;
        let draws_end = d.end(&Location::section("draws"), draws_start, draws_count, DRAW_FLOATS)?;
        let solids_start = draws_end;
        let solids_end = d.end(&Location::section("solids"), solids_start, solids_lanes, 1)?;
        let decals_start = solids_end;
        let decals_end = d.end(&Location::section("decals"), decals_start, decals_count, DECAL_FLOATS)?;
        let sounds_start = decals_end;
        let sounds_end = d.end(&Location::section("sounds"), sounds_start, sounds_count, SOUND_FLOATS)?;
        let sprites_start = sounds_end;
        let sprites_end = d.end(&Location::section("sprites"), sprites_start, sprites_count, SPRITE_FLOATS)?;
        let insets_start = sprites_end;
        let insets_end = d.end(&Location::section("insets"), insets_start, insets_count, INSET_FLOATS)?;
        let inset_draws_start = insets_end;
        let inset_draws_end = d.end(&Location::section("inset_draws"), inset_draws_start, inset_draws_count, DRAW_FLOATS)?;
        let mut inset_draws_cursor = inset_draws_start;
        let preferences_start = inset_draws_end;
        let preferences_end = d.end(&Location::section("preferences"), preferences_start, preferences_lanes, 1)?;
        let fields_start = preferences_end;
        let fields_end = d.end(&Location::section("fields"), fields_start, fields_lanes, 1)?;
        let ui_start = fields_end;
        let ui_end = d.end(&Location::section("ui"), ui_start, ui_lanes, 1)?;
        let services_start = ui_end;
        let services_end = d.end(&Location::section("services"), services_start, services_lanes, 1)?;
        let buses_start = services_end;
        let buses_end = d.end(&Location::section("buses"), buses_start, buses_count, BUS_FLOATS)?;
        let voices_start = buses_end;
        let voices_end = d.end(&Location::section("voices"), voices_start, voices_count, VOICE_FLOATS)?;
        let poses_start = voices_end;
        let poses_end = d.end(&Location::section("poses"), poses_start, poses_count, POSE_FLOATS)?;
        let emitters_start = poses_end;
        let emitters_end = d.end(&Location::section("emitters"), emitters_start, emitters_count, EMITTER_FLOATS)?;
        let lights_start = emitters_end;
        let lights_end = d.end(&Location::section("lights"), lights_start, lights_count, LIGHT_FLOATS)?;
        let listener_start = lights_end;
        let listener_end = d.end(&Location::section("listener"), listener_start, listener_count, LISTENER_FLOATS)?;
        let panners_start = listener_end;
        let panners_end = d.end(&Location::section("panners"), panners_start, panners_count, PANNER_FLOATS)?;
        let extras_start = panners_end;
        let extras_cursor = &mut { extras_start };
        d.count(&header.field("save_requested"), 12, 1)?;
        d.count(&header.field("load_requested"), 13, 1)?;
        d.count(&header.field("scene_save_requested"), 14, 1)?;
        d.count(&header.field("scene_load_requested"), 15, 1)?;
        d.count(&header.field("checkpoint_requested"), 16, 1)?;
        d.count(&header.field("undo_requested"), 17, 1)?;
        d.count(&header.field("redo_requested"), 18, 1)?;
        d.count(&header.field("exit_requested"), 19, 1)?;
        d.values(&header.field("view"), 20, 16, &[])?;
        d.values(&header.field("projection"), 36, 16, &[])?;
        d.values(&header.field("clear"), 52, 4, &["r","g","b","a"])?;
        d.values(&header.field("ambient"), 56, 3, &["x","y","z"])?;
        d.values(&header.field("light_direction"), 59, 3, &["x","y","z"])?;
        d.values(&header.field("light_color"), 62, 3, &["x","y","z"])?;
        d.values(&header.field("aberration"), 65, 2, &["x","y"])?;
        d.finite(&header.field("quality.msaa"), 67)?;
        d.count(&header.field("quality.fxaa"), 68, 1)?;
        d.finite(&header.field("quality.scale"), 69)?;
        let _ = d.count(&header.field("quality.filtering"), 70, 16)?;
        d.count(&header.field("quality.occlusion"), 71, 1)?;
        d.count(&header.field("quality.sky"), 72, 1)?;
        d.count(&header.field("quality.effects"), 73, 1)?;
        d.values(&header.field("sky.zenith"), 74, 3, &["x","y","z"])?;
        d.values(&header.field("sky.horizon"), 77, 3, &["x","y","z"])?;
        d.values(&header.field("sky.ground"), 80, 3, &["x","y","z"])?;
        d.values(&header.field("sky.sun"), 83, 3, &["x","y","z"])?;
        d.finite(&header.field("sky.night"), 86)?;
        d.finite(&header.field("fog.radius"), 87)?;
        d.finite(&header.field("fog.density"), 88)?;
        d.finite(&header.field("fog.height"), 89)?;
        d.values(&header.field("shadow.view"), 90, 16, &[])?;
        d.values(&header.field("shadow.projection"), 106, 16, &[])?;
        d.finite(&header.field("shadow.texel"), 122)?;
        let _ = d.count(&header.field("shadow.texels"), 123, 8192)?;
        d.finite(&header.field("blur"), 124)?;
        d.count(&header.field("quality.bloom"), 125, 1)?;
        d.finite(&header.field("clouds.cover"), 126)?;
        d.values(&header.field("clouds.drift"), 127, 2, &["x","z"])?;
        d.count(&header.field("hide_cursor"), 129, 1)?;
        d.count(&header.field("quality.taa"), 140, 1)?;
        d.count(&header.field("loading"), 144, 1)?;
        d.count(&header.field("show_captions"), 145, 1)?;
        d.values(&Location::section("state"), state_start, state_lanes, &[])?;
        let mut base = draws_start;
        for item in 0..draws_count {
            let path = &Location::section("draws").index(item);
            d.reference(&path.field("mesh"), base, catalog.meshes.len(), true)?;
            d.reference(&path.field("material"), base + 1, catalog.materials.len(), true)?;
            d.values(&path.field("transform"), base + 2, 16, &[])?;
            let count = d.count(&path.field("parameters").field("count"), base + 18, 4)?;
            if count != 0 && count != 4 { return Err(path.field("parameters").field("count").error(base + 18, "0 or 4", Actual::Value(count as f32))); }
            d.values(&path.field("parameters"), base + 18 + 1, 4, &[])?;
            let count = d.count(&path.field("extra"), base + 23, MAX_DRAW_EXTRA)?;
            let end = d.end(&path.field("extra"), *extras_cursor, count, 4)?;
            d.values(&path.field("extra"), *extras_cursor, end - *extras_cursor, &[])?;
            *extras_cursor = end;
            base += DRAW_FLOATS;
        }
        d.exhausted(&Location::section("draws"), base, draws_end)?;
        let mut base = solids_start;
        for item in 0..solids_count {
            let path = &Location::section("solids").index(item);
            d.within(path, base, d.end(path, base, 1, 1)?, solids_end)?;
            let faces_count = d.count(&path.field("faces.count"), base, MAX_SOLID_FACES)?;
            if faces_count < 4 { return Err(path.field("faces.count").error(base, "at least 4", Actual::Value(faces_count as f32))); }
            let first = base + 1;
            let end = d.end(&path.field("faces"), first, faces_count, SOLID_FACE_FLOATS)?;
            d.within(&path.field("faces"), first, end, solids_end)?;
            for child in 0..faces_count {
                inspect_solid_face(d, catalog, first + child * SOLID_FACE_FLOATS, &path.field("faces").index(child), extras_cursor)?;
            }
            base = end;
        }
        d.exhausted(&Location::section("solids"), base, solids_end)?;
        let mut base = decals_start;
        for item in 0..decals_count {
            let path = &Location::section("decals").index(item);
            d.reference(&path.field("material"), base, catalog.materials.len(), true)?;
            d.values(&path.field("transform"), base + 1, 16, &[])?;
            d.finite(&path.field("opacity"), base + 17)?;
            base += DECAL_FLOATS;
        }
        d.exhausted(&Location::section("decals"), base, decals_end)?;
        let mut base = sounds_start;
        for item in 0..sounds_count {
            let path = &Location::section("sounds").index(item);
            d.reference(&path.field("clip"), base, catalog.sounds.len(), true)?;
            let bus = d.count(&path.field("bus"), base + 1, 15)?;
            d.finite(&path.field("gain"), base + 2)?;
            d.finite(&path.field("pan"), base + 3)?;
            d.finite(&path.field("pitch"), base + 4)?;
            d.finite(&path.field("lowpass"), base + 5)?;
            d.finite(&path.field("reverb"), base + 6)?;
            base += SOUND_FLOATS;
        }
        d.exhausted(&Location::section("sounds"), base, sounds_end)?;
        let mut base = sprites_start;
        for item in 0..sprites_count {
            let path = &Location::section("sprites").index(item);
            d.finite(&path.field("x"), base)?;
            d.finite(&path.field("y"), base + 1)?;
            d.finite(&path.field("width"), base + 2)?;
            d.finite(&path.field("height"), base + 3)?;
            d.finite(&path.field("u0"), base + 4)?;
            d.finite(&path.field("v0"), base + 5)?;
            d.finite(&path.field("u1"), base + 6)?;
            d.finite(&path.field("v1"), base + 7)?;
            d.values(&path.field("color"), base + 8, 4, &[])?;
            d.reference(&path.field("material"), base + 12, catalog.materials.len(), true)?;
            d.values(&path.field("parameters"), base + 13, 4, &[])?;
            base += SPRITE_FLOATS;
        }
        d.exhausted(&Location::section("sprites"), base, sprites_end)?;
        let mut base = insets_start;
        for item in 0..insets_count {
            let path = &Location::section("insets").index(item);
            d.finite(&path.field("x"), base)?;
            d.finite(&path.field("y"), base + 1)?;
            d.finite(&path.field("width"), base + 2)?;
            d.finite(&path.field("height"), base + 3)?;
            d.values(&path.field("view"), base + 4, 16, &[])?;
            d.values(&path.field("projection"), base + 20, 16, &[])?;
            d.values(&path.field("ambient"), base + 36, 3, &["x","y","z"])?;
            d.values(&path.field("light_direction"), base + 39, 3, &["x","y","z"])?;
            d.values(&path.field("light_color"), base + 42, 3, &["x","y","z"])?;
            let count = d.count(&path.field("draws"), base + 45, MAX_DRAWS)?;
            let end = d.end(&path.field("draws"), inset_draws_cursor, count, DRAW_FLOATS)?;
            d.within(&path.field("draws"), inset_draws_cursor, end, inset_draws_end)?;
            for child in 0..count {
                inspect_draw(d, catalog, inset_draws_cursor, &path.field("draws").index(child), extras_cursor)?;
                inset_draws_cursor += DRAW_FLOATS;
            }
            d.finite(&path.field("pixel_size"), base + 46)?;
            let after_sprites = d.count(&path.field("after_sprites"), base + 47, MAX_SPRITES)?;
            base += INSET_FLOATS;
        }
        d.exhausted(&Location::section("insets"), base, insets_end)?;
        d.exhausted(&Location::section("inset_draws"), inset_draws_cursor, inset_draws_end)?;
        d.values(&Location::section("preferences"), preferences_start, preferences_lanes, &[])?;
        let mut base = fields_start;
        for item in 0..fields_count {
            let path = &Location::section("fields").index(item);
            d.within(path, base, d.end(path, base, 1, 7)?, fields_end)?;
            d.reference(&path.field("material"), base, catalog.materials.len(), true)?;
            let width = d.count(&path.field("width"), base + 1, MAX_FIELD)?;
            let height = d.count(&path.field("height"), base + 2, MAX_FIELD)?;
            let x = d.count(&path.field("x"), base + 3, width)?;
            let z = d.count(&path.field("z"), base + 4, height)?;
            let columns = d.count(&path.field("columns"), base + 5, width - x)?;
            let rows = d.count(&path.field("rows"), base + 6, height - z)?;
            let first = base + 7;
            let end = d.end(&path.field("values"), first, columns * rows, 1)?;
            d.within(&path.field("values"), first, end, fields_end)?;
            d.values(&path.field("values"), first, end - first, &[])?;
            base = end;
        }
        d.exhausted(&Location::section("fields"), base, fields_end)?;
        let mut base = ui_start;
        for item in 0..ui_count {
            let path = &Location::section("ui").index(item);
            d.within(path, base, d.end(path, base, 1, 24)?, ui_end)?;
            let id = d.count(&path.field("id"), base, MAX_UI_ID)?;
            let parent = d.count(&path.field("parent"), base + 1, MAX_UI_NODES)?;
            let role = d.count(&path.field("role"), base + 2, 23)?;
            d.finite(&path.field("x"), base + 3)?;
            d.finite(&path.field("y"), base + 4)?;
            d.finite(&path.field("width"), base + 5)?;
            d.finite(&path.field("height"), base + 6)?;
            d.count(&path.field("focusable"), base + 7, 1)?;
            d.count(&path.field("focused"), base + 8, 1)?;
            d.count(&path.field("disabled"), base + 9, 1)?;
            d.count(&path.field("hidden"), base + 10, 1)?;
            d.count(&path.field("modal"), base + 11, 1)?;
            let checked = d.count(&path.field("checked"), base + 12, 3)?;
            let expanded = d.count(&path.field("expanded"), base + 13, 2)?;
            let selected = d.count(&path.field("selected"), base + 14, 2)?;
            let live = d.count(&path.field("live"), base + 15, 2)?;
            d.finite(&path.field("value"), base + 16)?;
            d.finite(&path.field("minimum"), base + 17)?;
            d.finite(&path.field("maximum"), base + 18)?;
            d.finite(&path.field("step"), base + 19)?;
            let actions = d.count(&path.field("actions"), base + 20, 255)?;
            let text_length = d.count(&path.field("text_length"), base + 21, MAX_UI_TEXT)?;
            let name_length = d.count(&path.field("name_length"), base + 22, text_length)?;
            let description_length = d.count(&path.field("description_length"), base + 23, text_length - name_length)?;
            let first = base + 24;
            let end = d.end(&path.field("text"), first, text_length, 1)?;
            d.within(&path.field("text"), first, end, ui_end)?;
            d.values(&path.field("text"), first, end - first, &[])?;
            base = end;
        }
        d.exhausted(&Location::section("ui"), base, ui_end)?;
        let mut base = services_start;
        for item in 0..services_count {
            let path = &Location::section("services").index(item);
            d.within(path, base, d.end(path, base, 1, 8)?, services_end)?;
            let request = d.count(&path.field("request"), base, MAX_SERVICE_ID)?;
            let service = d.count(&path.field("service"), base + 1, 14)?;
            let operation = d.count(&path.field("operation"), base + 2, MAX_SERVICE_OPERATION)?;
            d.finite(&path.field("a"), base + 3)?;
            d.finite(&path.field("b"), base + 4)?;
            d.finite(&path.field("c"), base + 5)?;
            d.finite(&path.field("d"), base + 6)?;
            let value_count = d.count(&path.field("value_count"), base + 7, MAX_SERVICE_DATA)?;
            let first = base + 8;
            let end = d.end(&path.field("values"), first, value_count, 1)?;
            d.within(&path.field("values"), first, end, services_end)?;
            d.values(&path.field("values"), first, end - first, &[])?;
            base = end;
        }
        d.exhausted(&Location::section("services"), base, services_end)?;
        let mut base = buses_start;
        for item in 0..buses_count {
            let path = &Location::section("buses").index(item);
            d.finite(&path.field("gain"), base)?;
            d.finite(&path.field("lowpass"), base + 1)?;
            d.finite(&path.field("reverb"), base + 2)?;
            base += BUS_FLOATS;
        }
        d.exhausted(&Location::section("buses"), base, buses_end)?;
        let mut base = voices_start;
        for item in 0..voices_count {
            let path = &Location::section("voices").index(item);
            let id = d.count(&path.field("id"), base, MAX_VOICE_ID)?;
            d.reference(&path.field("clip"), base + 1, catalog.sounds.len(), true)?;
            let bus = d.count(&path.field("bus"), base + 2, 15)?;
            d.finite(&path.field("gain"), base + 3)?;
            d.finite(&path.field("pan"), base + 4)?;
            d.finite(&path.field("pitch"), base + 5)?;
            d.finite(&path.field("lowpass"), base + 6)?;
            d.finite(&path.field("reverb"), base + 7)?;
            d.count(&path.field("looping"), base + 8, 1)?;
            d.finite(&path.field("start"), base + 9)?;
            d.finite(&path.field("fade_in"), base + 10)?;
            d.finite(&path.field("fade_out"), base + 11)?;
            base += VOICE_FLOATS;
        }
        d.exhausted(&Location::section("voices"), base, voices_end)?;
        let mut base = poses_start;
        for item in 0..poses_count {
            let path = &Location::section("poses").index(item);
            let draw = d.count(&path.field("draw"), base, MAX_DRAWS)?;
            let clip = d.count(&path.field("clip"), base + 1, 65535)?;
            d.finite(&path.field("time"), base + 2)?;
            d.finite(&path.field("weight"), base + 3)?;
            base += POSE_FLOATS;
        }
        d.exhausted(&Location::section("poses"), base, poses_end)?;
        let mut base = emitters_start;
        for item in 0..emitters_count {
            let path = &Location::section("emitters").index(item);
            d.reference(&path.field("material"), base, catalog.materials.len(), true)?;
            d.values(&path.field("transform"), base + 1, 16, &[])?;
            let seed = d.count(&path.field("seed"), base + 17, 16777215)?;
            d.finite(&path.field("time"), base + 18)?;
            d.finite(&path.field("duration"), base + 19)?;
            d.finite(&path.field("rate"), base + 20)?;
            d.finite(&path.field("lifetime"), base + 21)?;
            d.finite(&path.field("radius"), base + 22)?;
            d.values(&path.field("velocity"), base + 23, 3, &["x","y","z"])?;
            d.finite(&path.field("spread"), base + 26)?;
            d.values(&path.field("gravity"), base + 27, 3, &["x","y","z"])?;
            d.finite(&path.field("drag"), base + 30)?;
            d.finite(&path.field("size_start"), base + 31)?;
            d.finite(&path.field("size_end"), base + 32)?;
            d.values(&path.field("color_start"), base + 33, 4, &["r","g","b","a"])?;
            d.values(&path.field("color_end"), base + 37, 4, &["r","g","b","a"])?;
            d.count(&path.field("additive"), base + 41, 1)?;
            base += EMITTER_FLOATS;
        }
        d.exhausted(&Location::section("emitters"), base, emitters_end)?;
        let mut base = lights_start;
        for item in 0..lights_count {
            let path = &Location::section("lights").index(item);
            d.values(&path.field("position"), base, 3, &["x","y","z"])?;
            d.finite(&path.field("radius"), base + 3)?;
            d.values(&path.field("color"), base + 4, 3, &["x","y","z"])?;
            d.values(&path.field("direction"), base + 7, 3, &["x","y","z"])?;
            d.finite(&path.field("inner"), base + 10)?;
            d.finite(&path.field("outer"), base + 11)?;
            base += LIGHT_FLOATS;
        }
        d.exhausted(&Location::section("lights"), base, lights_end)?;
        let mut base = listener_start;
        for item in 0..listener_count {
            let path = &Location::section("listener").index(item);
            d.values(&path.field("position"), base, 3, &["x","y","z"])?;
            d.values(&path.field("forward"), base + 3, 3, &["x","y","z"])?;
            d.values(&path.field("up"), base + 6, 3, &["x","y","z"])?;
            base += LISTENER_FLOATS;
        }
        d.exhausted(&Location::section("listener"), base, listener_end)?;
        let mut base = panners_start;
        for item in 0..panners_count {
            let path = &Location::section("panners").index(item);
            let voice = d.count(&path.field("voice"), base, MAX_VOICE_ID)?;
            let sound = d.count(&path.field("sound"), base + 1, 255)?;
            d.values(&path.field("position"), base + 2, 3, &["x","y","z"])?;
            d.values(&path.field("orientation"), base + 5, 3, &["x","y","z"])?;
            let distance_model = d.count(&path.field("distance_model"), base + 8, 2)?;
            d.finite(&path.field("ref_distance"), base + 9)?;
            d.finite(&path.field("max_distance"), base + 10)?;
            d.finite(&path.field("rolloff"), base + 11)?;
            d.finite(&path.field("cone_inner"), base + 12)?;
            d.finite(&path.field("cone_outer"), base + 13)?;
            d.finite(&path.field("cone_outer_gain"), base + 14)?;
            base += PANNER_FLOATS;
        }
        d.exhausted(&Location::section("panners"), base, panners_end)?;
        inspect_extensions(d, *extras_cursor)?;
        Ok(())
    }
    #[cold]
    fn explain_frame(packet: &[f32], schema: u32, catalog: &Catalog) -> FrameDiagnostic {
        let mut diagnostic = inspect_frame(packet, schema, catalog).err().unwrap_or_else(|| Location::section("packet").error(0, "a valid render packet (structural context unavailable)", Actual::Length(packet.len())));
        if diagnostic.section != "extensions" && let Ok(start) = provenance_start(packet, schema) {
            diagnostic.origin = crate::diagnostic::origin(packet, start, diagnostic.lane).map(Box::new);
        }
        diagnostic
    }
    #[allow(unused_variables, unused_mut)]
    fn provenance_start(packet: &[f32], schema: u32) -> Check<usize> {
        let d = &Inspector { packet };
        let header = Location::section("header");
        d.equal(&header.field("magic"), 0, PACKET_MAGIC)?;
        d.equal(&header.field("schema"), 1, schema as f32)?;
        let state_lanes = d.count(&header.field("state_count"), 2, MAX_STATE_FLOATS)?;
        let draws_count = d.count(&header.field("draw_count"), 3, MAX_DRAWS)?;
        let solids_count = d.count(&header.field("solid_count"), 4, MAX_SOLIDS)?;
        let solids_lanes = d.count(&header.field("solid_lanes"), 5, 32100000)?;
        let decals_count = d.count(&header.field("decal_count"), 6, MAX_DECALS)?;
        let sounds_count = d.count(&header.field("sound_count"), 7, MAX_SOUNDS)?;
        let sprites_count = d.count(&header.field("sprite_count"), 8, MAX_SPRITES)?;
        let insets_count = d.count(&header.field("inset_count"), 9, MAX_INSETS)?;
        let inset_draws_count = d.count(&header.field("inset_draw_count"), 10, MAX_DRAWS)?;
        let preferences_lanes = d.count(&header.field("preference_count"), 11, 256)?;
        let fields_count = d.count(&header.field("field_count"), 130, MAX_FIELD_PATCHES)?;
        let fields_lanes = d.count(&header.field("field_lanes"), 131, 16805888)?;
        let ui_count = d.count(&header.field("ui_count"), 132, MAX_UI_NODES)?;
        let ui_lanes = d.count(&header.field("ui_lanes"), 133, 4292608)?;
        let services_count = d.count(&header.field("service_count"), 134, MAX_SERVICE_REQUESTS)?;
        let services_lanes = d.count(&header.field("service_lanes"), 135, 4196352)?;
        let buses_count = d.count(&header.field("bus_count"), 136, MAX_BUSES)?;
        let voices_count = d.count(&header.field("voice_count"), 137, MAX_VOICES)?;
        let poses_count = d.count(&header.field("pose_count"), 138, MAX_POSES)?;
        let emitters_count = d.count(&header.field("emitter_count"), 139, MAX_EMITTERS)?;
        let lights_count = d.count(&header.field("light_count"), 141, MAX_LIGHTS)?;
        let listener_count = d.count(&header.field("listener_count"), 142, 1)?;
        let panners_count = d.count(&header.field("panner_count"), 143, MAX_PANNERS)?;
        let state_start = PACKET_HEADER_FLOATS;
        let state_end = d.end(&Location::section("state"), state_start, state_lanes, 1)?;
        let draws_start = state_end;
        let draws_end = d.end(&Location::section("draws"), draws_start, draws_count, DRAW_FLOATS)?;
        let solids_start = draws_end;
        let solids_end = d.end(&Location::section("solids"), solids_start, solids_lanes, 1)?;
        let decals_start = solids_end;
        let decals_end = d.end(&Location::section("decals"), decals_start, decals_count, DECAL_FLOATS)?;
        let sounds_start = decals_end;
        let sounds_end = d.end(&Location::section("sounds"), sounds_start, sounds_count, SOUND_FLOATS)?;
        let sprites_start = sounds_end;
        let sprites_end = d.end(&Location::section("sprites"), sprites_start, sprites_count, SPRITE_FLOATS)?;
        let insets_start = sprites_end;
        let insets_end = d.end(&Location::section("insets"), insets_start, insets_count, INSET_FLOATS)?;
        let inset_draws_start = insets_end;
        let inset_draws_end = d.end(&Location::section("inset_draws"), inset_draws_start, inset_draws_count, DRAW_FLOATS)?;
        let mut inset_draws_cursor = inset_draws_start;
        let preferences_start = inset_draws_end;
        let preferences_end = d.end(&Location::section("preferences"), preferences_start, preferences_lanes, 1)?;
        let fields_start = preferences_end;
        let fields_end = d.end(&Location::section("fields"), fields_start, fields_lanes, 1)?;
        let ui_start = fields_end;
        let ui_end = d.end(&Location::section("ui"), ui_start, ui_lanes, 1)?;
        let services_start = ui_end;
        let services_end = d.end(&Location::section("services"), services_start, services_lanes, 1)?;
        let buses_start = services_end;
        let buses_end = d.end(&Location::section("buses"), buses_start, buses_count, BUS_FLOATS)?;
        let voices_start = buses_end;
        let voices_end = d.end(&Location::section("voices"), voices_start, voices_count, VOICE_FLOATS)?;
        let poses_start = voices_end;
        let poses_end = d.end(&Location::section("poses"), poses_start, poses_count, POSE_FLOATS)?;
        let emitters_start = poses_end;
        let emitters_end = d.end(&Location::section("emitters"), emitters_start, emitters_count, EMITTER_FLOATS)?;
        let lights_start = emitters_end;
        let lights_end = d.end(&Location::section("lights"), lights_start, lights_count, LIGHT_FLOATS)?;
        let listener_start = lights_end;
        let listener_end = d.end(&Location::section("listener"), listener_start, listener_count, LISTENER_FLOATS)?;
        let panners_start = listener_end;
        let panners_end = d.end(&Location::section("panners"), panners_start, panners_count, PANNER_FLOATS)?;
        let extras_start = panners_end;
        let mut end = extras_start;
        for item in 0..draws_count {
            let count = d.count(&Location::section("draws").index(item).field("extra"), draws_start + item * DRAW_FLOATS + 23, MAX_DRAW_EXTRA)?;
            end = d.end(&Location::section("extras"), end, count, 4)?;
        }
        for item in 0..inset_draws_count {
            let count = d.count(&Location::section("inset_draws").index(item).field("extra"), inset_draws_start + item * DRAW_FLOATS + 23, MAX_DRAW_EXTRA)?;
            end = d.end(&Location::section("extras"), end, count, 4)?;
        }
        Ok(end)
    }
    fn inspect_extensions(d: &Inspector<'_>, mut at: usize) -> Check {
        let start = at;
        let path = &Location::section("extensions");
        let mut provenance_seen = false;
        if at < d.packet.len() {
            d.equal(&path.field("marker"), at, EXTENSIONS_MARKER)?;
            let count = d.count(&path.field("count"), at + 1, MAX_EXTENSIONS)?;
            at += 2;
            for item in 0..count {
                let path = &path.index(item);
                let tag = d.count(&path.field("tag"), at, MAX_EXTENSION_TAG)?;
                let length = d.count(&path.field("length"), at + 1, if tag == PROVENANCE_TAG { MAX_PROVENANCE_LANES } else { MAX_EXTENSION_LANES })?;
                d.values(&path.field("values"), at + 2, length, &[])?;
                if tag == PROVENANCE_TAG {
                    if provenance_seen || crate::diagnostic::read_provenance(&d.packet[at + 2..at + 2 + length], start).is_err() {
                        return Err(path.field("provenance").error(at + 2, "one bounded version-1 provenance JSON object", Actual::Length(length)));
                    }
                    provenance_seen = true;
                }
                at = d.end(path, at + 2, length, 1)?;
            }
        }
        d.exhausted(path, at, d.packet.len())
    }

    /// The walk of an input packet's values that pen's tests compare with the game's
    /// (`pen/protocol_canonical.blot`): elapsed, the viewport's width and height, the events'
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

    /// The walk of a render packet's values that pen's tests compare with the game's
    /// (`pen/protocol_canonical.blot`): the header's values and then the sections, each in the
    /// schema's order, as described in pen-codegen's canonical.rs. It takes the schema and
    /// the state, which the frame does not hold, where the packet has them.
    pub fn canonical_render(frame: &RenderFrame, schema: u32, state: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
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
            frame.sky.zenith.x, frame.sky.zenith.y, frame.sky.zenith.z, frame.sky.horizon.x, frame.sky.horizon.y,
            frame.sky.horizon.z, frame.sky.ground.x, frame.sky.ground.y, frame.sky.ground.z, frame.sky.sun.x,
            frame.sky.sun.y, frame.sky.sun.z, frame.sky.night, frame.fog.radius, frame.fog.density,
            frame.fog.height,
        ]);
        out.extend_from_slice(&frame.shadow.view);
        out.extend_from_slice(&frame.shadow.projection);
        out.extend([
            frame.shadow.texel, frame.shadow.texels as f32, frame.blur, frame.clouds.cover, frame.clouds.drift.x,
            frame.clouds.drift.z, f32::from(frame.hide_cursor), f32::from(frame.loading),
            f32::from(frame.show_captions), state.len() as f32,
        ]);
        out.extend_from_slice(state);
        out.push(frame.draws.len() as f32);
        for draw in &frame.draws {
            canonical_draw(&mut out, draw);
        }
        out.push(frame.solids.len() as f32);
        for solid in &frame.solids {
            canonical_solid(&mut out, solid);
        }
        out.push(frame.decals.len() as f32);
        for decal in &frame.decals {
            canonical_decal(&mut out, decal);
        }
        out.push(frame.sounds.len() as f32);
        for sound in &frame.sounds {
            canonical_sound(&mut out, sound);
        }
        out.push(frame.sprites.len() as f32);
        for sprite in &frame.sprites {
            canonical_sprite(&mut out, sprite);
        }
        out.push(frame.insets.len() as f32);
        for inset in &frame.insets {
            canonical_inset(&mut out, inset);
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
            canonical_field_patch(&mut out, field_patch);
        }
        out.push(frame.ui.len() as f32);
        for ui_node in &frame.ui {
            canonical_ui_node(&mut out, ui_node);
        }
        out.push(frame.services.len() as f32);
        for service_request in &frame.services {
            canonical_service_request(&mut out, service_request);
        }
        out.push(frame.buses.len() as f32);
        for bus in &frame.buses {
            canonical_bus(&mut out, bus);
        }
        out.push(frame.voices.len() as f32);
        for voice in &frame.voices {
            canonical_voice(&mut out, voice);
        }
        out.push(frame.poses.len() as f32);
        for pose in &frame.poses {
            canonical_pose(&mut out, pose);
        }
        out.push(frame.emitters.len() as f32);
        for emitter in &frame.emitters {
            canonical_emitter(&mut out, emitter);
        }
        out.push(frame.lights.len() as f32);
        for light in &frame.lights {
            canonical_light(&mut out, light);
        }
        out.push(frame.listener.len() as f32);
        for listener in &frame.listener {
            canonical_listener(&mut out, listener);
        }
        out.push(frame.panners.len() as f32);
        for panner in &frame.panners {
            canonical_panner(&mut out, panner);
        }
        out
    }

    fn canonical_draw(out: &mut Vec<f32>, draw: &Draw) {
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

    fn canonical_solid_face(out: &mut Vec<f32>, solid_face: &SolidFace) {
        out.extend([
            solid_face.normal.x, solid_face.normal.y, solid_face.normal.z, solid_face.distance,
            solid_face.material as f32,
        ]);
    }

    fn canonical_solid(out: &mut Vec<f32>, solid: &Solid) {
        out.push(solid.faces.len() as f32);
        for solid_face in &solid.faces {
            canonical_solid_face(out, solid_face);
        }
    }

    fn canonical_decal(out: &mut Vec<f32>, decal: &Decal) {
        out.push(decal.material as f32);
        out.extend_from_slice(&decal.transform);
        out.push(decal.opacity);
    }

    fn canonical_sound(out: &mut Vec<f32>, sound: &Sound) {
        out.extend([
            sound.clip as f32, sound.bus as f32, sound.gain, sound.pan, sound.pitch, sound.lowpass, sound.reverb,
        ]);
    }

    fn canonical_sprite(out: &mut Vec<f32>, sprite: &Sprite) {
        out.extend([
            sprite.x, sprite.y, sprite.width, sprite.height, sprite.u0, sprite.v0, sprite.u1, sprite.v1,
        ]);
        out.extend_from_slice(&sprite.color);
        out.push(sprite.material as f32);
        out.extend_from_slice(&sprite.parameters);
    }

    fn canonical_inset(out: &mut Vec<f32>, inset: &Inset) {
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

    fn canonical_field_patch(out: &mut Vec<f32>, field_patch: &FieldPatch) {
        out.extend([
            field_patch.material as f32, field_patch.width as f32, field_patch.height as f32,
            field_patch.x as f32, field_patch.z as f32, field_patch.columns as f32, field_patch.rows as f32,
            field_patch.values.len() as f32,
        ]);
        out.extend_from_slice(&field_patch.values);
    }

    fn canonical_ui_node(out: &mut Vec<f32>, ui_node: &UiNode) {
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

    fn canonical_service_request(out: &mut Vec<f32>, service_request: &ServiceRequest) {
        out.extend([
            service_request.request as f32, service_request.service as f32, service_request.operation as f32,
            service_request.a, service_request.b, service_request.c, service_request.d,
            service_request.value_count as f32, service_request.values.len() as f32,
        ]);
        out.extend_from_slice(&service_request.values);
    }

    fn canonical_bus(out: &mut Vec<f32>, bus: &Bus) {
        out.extend([bus.gain, bus.lowpass, bus.reverb]);
    }

    fn canonical_voice(out: &mut Vec<f32>, voice: &Voice) {
        out.extend([
            voice.id as f32, voice.clip as f32, voice.bus as f32, voice.gain, voice.pan, voice.pitch,
            voice.lowpass, voice.reverb, f32::from(voice.looping), voice.start, voice.fade_in, voice.fade_out,
        ]);
    }

    fn canonical_pose(out: &mut Vec<f32>, pose: &Pose) {
        out.extend([pose.draw as f32, pose.clip as f32, pose.time, pose.weight]);
    }

    fn canonical_emitter(out: &mut Vec<f32>, emitter: &Emitter) {
        out.push(emitter.material as f32);
        out.extend_from_slice(&emitter.transform);
        out.extend([
            emitter.seed as f32, emitter.time, emitter.duration, emitter.rate, emitter.lifetime, emitter.radius,
            emitter.velocity.x, emitter.velocity.y, emitter.velocity.z, emitter.spread, emitter.gravity.x,
            emitter.gravity.y, emitter.gravity.z, emitter.drag, emitter.size_start, emitter.size_end,
            emitter.color_start.r, emitter.color_start.g, emitter.color_start.b, emitter.color_start.a,
            emitter.color_end.r, emitter.color_end.g, emitter.color_end.b, emitter.color_end.a,
            f32::from(emitter.additive),
        ]);
    }

    fn canonical_light(out: &mut Vec<f32>, light: &Light) {
        out.extend([
            light.position.x, light.position.y, light.position.z, light.radius, light.color.x, light.color.y,
            light.color.z, light.direction.x, light.direction.y, light.direction.z, light.inner, light.outer,
        ]);
    }

    fn canonical_listener(out: &mut Vec<f32>, listener: &Listener) {
        out.extend([
            listener.position.x, listener.position.y, listener.position.z, listener.forward.x,
            listener.forward.y, listener.forward.z, listener.up.x, listener.up.y, listener.up.z,
        ]);
    }

    fn canonical_panner(out: &mut Vec<f32>, panner: &Panner) {
        out.extend([
            panner.voice as f32, panner.sound as f32, panner.position.x, panner.position.y, panner.position.z,
            panner.orientation.x, panner.orientation.y, panner.orientation.z, panner.distance_model as f32,
            panner.ref_distance, panner.max_distance, panner.rolloff, panner.cone_inner, panner.cone_outer,
            panner.cone_outer_gain,
        ]);
    }
}
