//! The frame packet's meaning, as gdev's `protocol.ts` describes it. Asset
//! references are indices into the [`Catalog`]; `boundary` has checked each
//! one is in range.

/// A column-major 4x4 matrix, as WGSL and the packet lay it out.
pub type Matrix = [f32; 16];

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GameEvent {
    /// A key by its browser `KeyboardEvent.key` name ("a", "A", "ArrowLeft", "Shift", ...).
    KeyDown(String),
    KeyUp(String),
    FocusLost,
    /// A console command for the game: which, and its argument.
    Console { command: f32, argument: f32 },
    /// A stored preference, played back when the game starts.
    Preference { index: f32, value: f32 },
    PointerDown { button: u32, x: f32, y: f32 },
    PointerUp { button: u32, x: f32, y: f32 },
    PointerMove { x: f32, y: f32 },
    Wheel { x: f32, y: f32, delta: f32 },
    AssetReady { key: String, kind: String, revision: u64 },
    AssetFailed { key: String, kind: String, revision: u64, message: String },
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

#[derive(Clone, Debug, PartialEq)]
pub struct Draw {
    pub material: u32,
    pub mesh: u32,
    pub transform: Matrix,
    /// Optional four shader values, overriding the material's instance defaults.
    pub parameters: Option<[f32; 4]>,
    /// Up to 32 more vec4s, flattened, that the shader reads after the
    /// parameters; a draw with any is drawn alone.
    pub extra: Option<Vec<f32>>,
}

/// How the game wants this frame drawn (packages/preferences.blot).
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
}

/// The sky's colors and the way toward the sun or moon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sky {
    pub zenith: Vector,
    pub horizon: Vector,
    pub ground: Vector,
    pub sun: Vector,
    /// 1 at full night, when stars show.
    pub night: f32,
}

/// Mist beyond `radius` of the middle; density 0 for none.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fog {
    pub radius: f32,
    pub density: f32,
    pub height: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drift {
    pub x: f32,
    pub z: f32,
}

/// How much of the sky clouds cover, 0 for none, and how far the wind has carried them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clouds {
    pub cover: f32,
    pub drift: Drift,
}

/// The shadow map's view; `texels` 0 for no shadows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shadow {
    pub view: Matrix,
    pub projection: Matrix,
    /// A map texel's width in world units.
    pub texel: f32,
    pub texels: u32,
}

/// A rectangle over the screen, in pixels from its top left, drawn by its
/// material's sprite shader, which reads the four `parameters` as it likes.
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Debug, PartialEq)]
pub struct Inset {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    /// Block size in screen pixels, or zero for a smooth image.
    pub pixel_size: f32,
    pub view: Matrix,
    pub projection: Matrix,
    pub ambient: Vector,
    pub light_direction: Vector,
    pub light_color: Vector,
    pub draws: Vec<Draw>,
}

/// `material`'s texture laid onto the surfaces inside the box (the cube -1..1
/// under `transform`) along its y axis; `opacity` fades it.
#[derive(Clone, Debug, PartialEq)]
pub struct Decal {
    pub material: u32,
    pub transform: Matrix,
    pub opacity: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolidFace {
    pub normal: Vector,
    pub distance: f32,
    pub material: u32,
}

/// A convex solid: the points behind every face plane (normal · p <= distance).
#[derive(Clone, Debug, PartialEq)]
pub struct Solid {
    pub faces: Vec<SolidFace>,
}

/// A sound to play once at the gain, pan (-1 left to 1 right) and pitch (1 as recorded) given.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sound {
    pub clip: u32,
    pub gain: f32,
    pub pan: f32,
    pub pitch: f32,
}

/// The clip to keep looping and its gain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ambience {
    pub clip: u32,
    pub gain: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aberration {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RenderFrame {
    /// The frame supplies its own cursor and asks the host to hide the system one.
    pub hide_cursor: bool,
    pub ambient: Vector,
    /// How far red moves from green, in screen heights across and down, and
    /// blue as far the other way, on surfaces whose material asks for it.
    pub aberration: Aberration,
    /// How far the picture smears toward the middle of the screen (0 to 0.5).
    pub blur: f32,
    pub clear: Color,
    pub quality: Quality,
    pub sky: Sky,
    pub fog: Fog,
    pub clouds: Clouds,
    pub shadow: Shadow,
    pub draws: Vec<Draw>,
    pub decals: Vec<Decal>,
    pub solids: Vec<Solid>,
    pub sprites: Vec<Sprite>,
    pub insets: Vec<Inset>,
    pub light_color: Vector,
    pub light_direction: Vector,
    pub view: Matrix,
    pub projection: Matrix,
    pub sounds: Vec<Sound>,
    pub ambience: Option<Ambience>,
    /// The player's preferences to store, when they changed; the host plays
    /// them back as Preference events when the game starts.
    pub preferences: Option<Vec<f32>>,
    pub load_requested: bool,
    pub save_requested: bool,
    pub scene_load_requested: bool,
    pub scene_save_requested: bool,
    /// An edit finished: the host records the scene for undo.
    pub checkpoint_requested: bool,
    pub undo_requested: bool,
    pub redo_requested: bool,
    /// The game is over: the host stops it.
    pub exit_requested: bool,
}
