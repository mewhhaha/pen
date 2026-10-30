//! Gather, a tiny game written in Rust for pen: steer the square with the
//! arrow keys (or WASD) and gather the gems. It is here to show that pen's
//! guest boundary is not Blot's alone. Built for wasm32-unknown-unknown and
//! bundled by `src/bin/bundle.rs`, it speaks what a Blot game does: the
//! `blot:abi` manifest's `main: (Array F32 -> Array F32 ! {Foreign}) ->
//! Unit` with its array arena, the program protocol (ink's
//! `lib/pen/program.blot`) and pen's packets (`protocol/schema.json`,
//! through pen-protocol's own constants), so pen runs it unchanged.
//!
//! The game and its protocol are plain Rust, tested natively against pen's
//! packet decoder; only `guest.rs` (the Wasm exports) and `adapt.rs` (which
//! gives them the guest ABI's shape) depend on where it runs.
use pen_protocol::boundary::{
    EVENT_FLOATS, FRAME_HEADER_FLOATS, PACKET_HEADER_FLOATS, PACKET_MAGIC, SPRITE_FLOATS, event_tags, key_code, lanes,
};

#[cfg(not(target_arch = "wasm32"))]
pub mod adapt;
#[cfg(target_arch = "wasm32")]
mod guest;

/// What a program says first (ink's `program.blot`): this magic number, its
/// protocol version, its state schema and whether it keeps scenes.
pub const HANDSHAKE_MAGIC: f32 = 4670544.0;
pub const PROGRAM_VERSION: f32 = 2.0;
/// The game's state schema: the layout of [`World::lanes`].
pub const SCHEMA: f32 = 1.0;

/// The host's requests, by operation number (pen-runtime's `Operation`).
const STOP: f32 = 0.0;
const CREATE: f32 = 1.0;
const FRAME: f32 = 2.0;
const SNAPSHOT: f32 = 3.0;
const RESTORE: f32 = 4.0;
/// The request a program answers before any: its handshake.
const HELLO: f32 = 7.0;
/// What a program answers in place of a state it cannot take.
const REFUSED: [f32; 1] = [-1.0];

/// The material every sprite draws with: the catalog's first.
const FILL: f32 = 0.0;
const GEMS: usize = 3;
/// Field units (the field is 1 across) per second.
const SPEED: f32 = 0.45;
const PLAYER: f32 = 0.035;
const GEM: f32 = 0.025;
/// The longest step a frame takes, so a stall does not jump the player.
const LONGEST_STEP: f32 = 0.1;

/// Direction keys held, as bits.
const LEFT: u32 = 1;
const RIGHT: u32 = 2;
const UP: u32 = 4;
const DOWN: u32 = 8;

/// The whole game: everything later frames depend on, so a snapshot of it
/// restores the game exactly (time travel, reloads and saves rely on it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct World {
    /// The player's middle, in field units from the top left.
    pub x: f32,
    pub y: f32,
    pub score: u32,
    /// The gems' places come from this (xorshift32).
    seed: u32,
    gems: [(f32, f32); GEMS],
    held: u32,
    /// Seconds played, for the gems' pulse.
    time: f32,
}

impl World {
    pub fn new() -> World {
        let mut world =
            World { x: 0.5, y: 0.5, score: 0, seed: 0x2545_f491, gems: [(0.0, 0.0); GEMS], held: 0, time: 0.0 };
        for gem in 0..GEMS {
            world.gems[gem] = world.place();
        }
        world
    }

    /// The next place for a gem, away from the edges.
    fn place(&mut self) -> (f32, f32) {
        let mut next = || {
            self.seed ^= self.seed << 13;
            self.seed ^= self.seed >> 17;
            self.seed ^= self.seed << 5;
            0.1 + 0.8 * (self.seed >> 8) as f32 / (1 << 24) as f32
        };
        (next(), next())
    }

    /// The state lanes: a version, the player, the score, the seed in two
    /// 16-bit halves (a lane holds integers to 2^24), the gems, the keys
    /// held and the time.
    pub fn lanes(&self) -> Vec<f32> {
        let mut lanes = vec![1.0, self.x, self.y, self.score as f32];
        lanes.extend([(self.seed >> 16) as f32, (self.seed & 0xffff) as f32]);
        for (x, y) in self.gems {
            lanes.extend([x, y]);
        }
        lanes.extend([self.held as f32, self.time]);
        lanes
    }

    /// The world `lanes` holds, or None if they are not one.
    pub fn from_lanes(lanes: &[f32]) -> Option<World> {
        let whole =
            |lane: f32, below: u32| (lane.fract() == 0.0 && (0.0..below as f32).contains(&lane)).then_some(lane as u32);
        let [version, x, y, score, high, low, gems @ .., held, time] = lanes else { return None };
        if *version != 1.0 || gems.len() != GEMS * 2 || !lanes.iter().all(|lane| lane.is_finite()) {
            return None;
        }
        let mut world = World {
            x: *x,
            y: *y,
            score: whole(*score, 1 << 24)?,
            seed: whole(*high, 1 << 16)? << 16 | whole(*low, 1 << 16)?,
            gems: [(0.0, 0.0); GEMS],
            held: whole(*held, 16)?,
            time: *time,
        };
        for (gem, place) in world.gems.iter_mut().zip(gems.chunks_exact(2)) {
            *gem = (place[0], place[1]);
        }
        Some(world)
    }

    /// Runs a frame's input (pen-protocol's `encode_frame` layout: elapsed
    /// seconds, viewport, events) and draws the result.
    pub fn frame(&mut self, input: &[f32]) -> Vec<f32> {
        let [elapsed, width, height, events, ..] = *input else { return Vec::new() };
        let events = input.get(FRAME_HEADER_FLOATS..).unwrap_or(&[]).chunks_exact(EVENT_FLOATS).take(events as usize);
        for event in events {
            let tag = event[0] as u32;
            let key = direction(event[1]);
            match tag {
                event_tags::KEY_DOWN => self.held |= key,
                event_tags::KEY_UP => self.held &= !key,
                event_tags::FOCUS_LOST => self.held = 0,
                _ => {}
            }
        }
        self.step(elapsed.clamp(0.0, LONGEST_STEP));
        self.draw(width, height)
    }

    fn step(&mut self, seconds: f32) {
        self.time += seconds;
        let axis = |less: u32, more: u32| -> f32 {
            (if self.held & more != 0 { 1.0 } else { 0.0 }) - if self.held & less != 0 { 1.0 } else { 0.0 }
        };
        let (dx, dy) = (axis(LEFT, RIGHT), axis(UP, DOWN));
        let length = (dx * dx + dy * dy).sqrt();
        if length > 0.0 {
            self.x = (self.x + dx / length * SPEED * seconds).clamp(PLAYER, 1.0 - PLAYER);
            self.y = (self.y + dy / length * SPEED * seconds).clamp(PLAYER, 1.0 - PLAYER);
        }
        for gem in 0..GEMS {
            let (x, y) = self.gems[gem];
            if (x - self.x).hypot(y - self.y) < PLAYER + GEM {
                self.score += 1;
                self.gems[gem] = self.place();
            }
        }
    }

    /// The render packet: a dark page, the field, the gems, the player and a
    /// row of marks counting the score, all sprites.
    fn draw(&self, width: f32, height: f32) -> Vec<f32> {
        let side = width.min(height) * 0.9;
        let (left, top) = ((width - side) / 2.0, (height - side) / 2.0);
        let mut sprites = Vec::new();
        let mut square = |x: f32, y: f32, size: f32, color: [f32; 4], round: f32| {
            let mut sprite = [0.0; SPRITE_FLOATS];
            let size = size * side;
            sprite[lanes::sprite::X] = left + x * side - size / 2.0;
            sprite[lanes::sprite::Y] = top + y * side - size / 2.0;
            sprite[lanes::sprite::WIDTH] = size;
            sprite[lanes::sprite::HEIGHT] = size;
            sprite[lanes::sprite::U1] = 1.0;
            sprite[lanes::sprite::V1] = 1.0;
            sprite[lanes::sprite::COLOR..lanes::sprite::COLOR + 4].copy_from_slice(&color);
            sprite[lanes::sprite::MATERIAL] = FILL;
            sprite[lanes::sprite::PARAMETERS] = round;
            sprites.push(sprite);
        };
        square(0.5, 0.5, 1.0, [0.11, 0.14, 0.2, 1.0], 0.04);
        for (gem, (x, y)) in self.gems.iter().enumerate() {
            let pulse = 1.0 + 0.15 * (self.time * 4.0 + gem as f32 * 2.1).sin();
            square(*x, *y, GEM * 2.0 * pulse, [1.0, 0.78, 0.25, 1.0], 1.0);
        }
        square(self.x, self.y, PLAYER * 2.0, [0.3, 0.85, 0.95, 1.0], 0.3);
        for mark in 0..self.score.min(40) {
            square(0.04 + mark as f32 * 0.024, 0.04, 0.016, [0.95, 0.95, 0.95, 1.0], 1.0);
        }
        let mut packet = vec![0.0; PACKET_HEADER_FLOATS];
        packet[lanes::header::MAGIC] = PACKET_MAGIC;
        packet[lanes::header::SCHEMA] = SCHEMA;
        packet[lanes::header::SPRITE_COUNT] = sprites.len() as f32;
        // No 3D scene: identity views and projections.
        for matrix in [
            lanes::header::VIEW,
            lanes::header::PROJECTION,
            lanes::header::SHADOW_VIEW,
            lanes::header::SHADOW_PROJECTION,
        ] {
            for diagonal in [0, 5, 10, 15] {
                packet[matrix + diagonal] = 1.0;
            }
        }
        packet[lanes::header::CLEAR..lanes::header::CLEAR + 4].copy_from_slice(&[0.04, 0.05, 0.08, 1.0]);
        packet[lanes::header::LIGHT_DIRECTION + 1] = -1.0;
        packet[lanes::header::LIGHT_COLOR..lanes::header::LIGHT_COLOR + 3].copy_from_slice(&[1.0, 1.0, 1.0]);
        packet[lanes::header::QUALITY_MSAA] = 1.0;
        packet[lanes::header::QUALITY_SCALE] = 1.0;
        packet[lanes::header::QUALITY_FILTERING] = 1.0;
        packet.extend(sprites.iter().flatten());
        packet
    }
}

impl Default for World {
    fn default() -> World {
        World::new()
    }
}

/// The direction a key code steers, as a bit; 0 for any other key.
fn direction(code: f32) -> u32 {
    let is = |names: &[&str]| names.iter().any(|name| key_code(name) == code);
    if is(&["ArrowLeft", "a"]) {
        LEFT
    } else if is(&["ArrowRight", "d"]) {
        RIGHT
    } else if is(&["ArrowUp", "w"]) {
        UP
    } else if is(&["ArrowDown", "s"]) {
        DOWN
    } else {
        0
    }
}

/// The program: answers the host's requests until it says stop, as ink's
/// `program.run` does. Each request after the first says whether the host
/// published the preceding output (its first lane). Version 2 advances live
/// simulation before output publication; rejection never rolls it back.
/// `exchange` sends a response and waits for the next request.
pub fn run(mut exchange: impl FnMut(&[f32]) -> Vec<f32>) {
    let mut world: Option<World> = None;
    let mut request = vec![1.0, HELLO];
    loop {
        let operation = request.get(1).copied().unwrap_or(HELLO);
        let payload = request.get(2..).unwrap_or(&[]);
        let (candidate, response) = match operation {
            STOP => return,
            CREATE => (Some(World::new()), vec![CREATE]),
            FRAME => {
                // The host creates or restores the world before any frame.
                let mut next = world.unwrap_or_default();
                let packet = next.frame(payload);
                (Some(next), [&[FRAME][..], &packet].concat())
            }
            SNAPSHOT => (world, [&[SNAPSHOT][..], &world.unwrap_or_default().lanes()].concat()),
            RESTORE => match World::from_lanes(payload) {
                Some(restored) => (Some(restored), vec![RESTORE]),
                None => (world, REFUSED.to_vec()),
            },
            HELLO => (world, vec![HANDSHAKE_MAGIC, PROGRAM_VERSION, SCHEMA, 0.0]),
            // Scenes (the editor's) this game has none of.
            _ => (world, REFUSED.to_vec()),
        };
        let accepted = response != REFUSED;
        if accepted {
            world = candidate;
        }
        request = exchange(&response);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_protocol::boundary::{decode_frame, encode_frame};
    use pen_protocol::{Catalog, Frame, GameEvent, Viewport};

    fn catalog() -> Catalog {
        let json = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/catalog.json")).unwrap();
        pen_protocol::boundary::parse_catalog(&serde_json::from_str(&json).unwrap()).unwrap()
    }

    fn input(elapsed: f64, events: Vec<GameEvent>) -> Vec<f32> {
        let frame = Frame { timestamp: 0.0, viewport: Viewport { width: 960, height: 640 }, events };
        encode_frame(&frame, elapsed, &[]).unwrap()
    }

    /// Runs the program on `requests` (the host's, after the handshake) and
    /// returns its responses, the handshake first.
    fn converse(requests: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut requests = requests.into_iter();
        let mut responses = Vec::new();
        run(|response| {
            responses.push(response.to_vec());
            requests.next().unwrap_or_else(|| vec![1.0, STOP])
        });
        responses
    }

    fn request(commit: bool, operation: f32, payload: &[f32]) -> Vec<f32> {
        [&[if commit { 1.0 } else { 0.0 }, operation][..], payload].concat()
    }

    #[test]
    fn speaks_the_program_protocol_with_pens_packets() {
        let right = input(0.5, vec![GameEvent::KeyDown("ArrowRight".into())]);
        let responses =
            converse(vec![request(true, CREATE, &[]), request(true, FRAME, &right), request(true, SNAPSHOT, &[])]);
        assert_eq!(responses[0], [HANDSHAKE_MAGIC, PROGRAM_VERSION, SCHEMA, 0.0]);
        assert_eq!(responses[1], [CREATE]);
        // pen's own decoder takes the frame, and the player moved right.
        let frame = decode_frame(&responses[2][1..], 1, &catalog()).unwrap();
        assert_eq!(frame.render.sprites.len(), 1 + GEMS + 1);
        let snapshot = World::from_lanes(&responses[3][1..]).unwrap();
        assert!(snapshot.x > 0.5 && snapshot.y == 0.5 && snapshot.held == RIGHT, "{snapshot:?}");
    }

    #[test]
    fn rejected_frames_keep_live_simulation_mutations() {
        let right = input(0.5, vec![GameEvent::KeyDown("d".into())]);
        let responses = converse(vec![
            request(true, CREATE, &[]),
            request(true, FRAME, &right),
            // Reject presentation; the explicit snapshot still observes the step.
            request(false, SNAPSHOT, &[]),
        ]);
        let live = World::from_lanes(&responses[3][1..]).unwrap();
        assert!(live.x > World::new().x);
        assert_eq!(live.held, RIGHT);
    }

    #[test]
    fn snapshots_restore_exactly_and_bad_states_are_refused() {
        let mut world = World::new();
        world.held = DOWN | LEFT;
        for _ in 0..90 {
            world.frame(&input(1.0 / 60.0, Vec::new()));
        }
        assert_eq!(World::from_lanes(&world.lanes()), Some(world));
        let responses = converse(vec![
            request(true, RESTORE, &world.lanes()),
            request(true, SNAPSHOT, &[]),
            request(true, RESTORE, &[2.0, 0.5]),
        ]);
        assert_eq!(responses[1], [RESTORE]);
        assert_eq!(responses[2][1..], world.lanes());
        assert_eq!(responses[3], REFUSED);
    }

    #[test]
    fn gathering_a_gem_scores_and_moves_it() {
        let mut world = World::new();
        let (x, y) = world.gems[0];
        world.x = x;
        world.y = y;
        world.step(0.0);
        assert_eq!(world.score, 1);
        assert_ne!(world.gems[0], (x, y));
        // The same input gives the same game, bit for bit.
        let mut again = World::new();
        again.x = x;
        again.y = y;
        again.step(0.0);
        assert_eq!(again, world);
    }
}
