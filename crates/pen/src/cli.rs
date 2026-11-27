//! Command-line argument specifications, parsing, usage text, and suggestions.
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::{mods, script};

/// A command: its name, the words it takes besides options (`[WORD]`
/// optional, `WORD...` any number), whether it works on a checkout, and
/// what it does.
struct CommandSpec {
    name: &'static str,
    operands: &'static str,
    #[cfg(feature = "compiler")]
    checkout: bool,
    about: &'static str,
}

/// Where the usage tells of an option.
#[derive(PartialEq)]
enum Place {
    /// Among the options for playing.
    Play,
    /// Under each of its commands.
    Command,
    /// In the usage's own words.
    Prose,
}

/// An option: its name, the value it takes ("" for a flag), the commands it
/// is for ("" plays a packaged game, without a command; none, any command),
/// where the usage tells of it, and what it does.
struct OptionSpec {
    name: &'static str,
    value: &'static str,
    commands: &'static [&'static str],
    place: Place,
    about: &'static str,
}

#[cfg(feature = "compiler")]
const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "tool",
        operands: "NAME [ARGS...]",
        checkout: true,
        about: "run a checkout-owned Rust tool from pen.json; put its arguments after --",
    },
    CommandSpec {
        name: "dev",
        operands: "",
        checkout: true,
        about: "compile the game's Blot program and play it, reloading the program, shaders and assets as they \
                change with the game's state kept, and rerunning the checkout's asset recipes (pipeline.json) \
                as their sources change",
    },
    CommandSpec {
        name: "web",
        operands: "",
        checkout: true,
        about: "browser development server, reloading as the program and assets change, with the dev tools",
    },
    CommandSpec { name: "run", operands: "", checkout: true, about: "compile once and play" },
    CommandSpec {
        name: "build",
        operands: "",
        checkout: true,
        about: "a shippable folder: the player runtime, its libraries and the game (named as pen.json's \
                `game` says), precompiled for the runtime's platform and a CPU baseline, with the assets its \
                catalog reaches, mesh JSON in binary",
    },
    CommandSpec {
        name: "check",
        operands: "",
        checkout: true,
        about: "compile the game and the dev tools, and lint the assets: files the runtime would refuse or that \
                are missing, unused, oversized or without the mip levels they need (pen.json's assets.lints)",
    },
    CommandSpec {
        name: "assets",
        operands: "",
        checkout: true,
        about: "bring the assets pipeline.json's steps make up to date with their sources: a step runs when its \
                sources changed, unless the asset cache holds what it built from the same sources before, which \
                is copied back",
    },
    CommandSpec {
        name: "codegen",
        operands: "",
        checkout: true,
        about: "generate or check protocol code from the embedded schema",
    },
    CommandSpec {
        name: "export",
        operands: "",
        checkout: true,
        about: "export game.wasm, the scene and the assets for verification",
    },
    CommandSpec {
        name: "test",
        operands: "[FILE]",
        checkout: true,
        about: "golden-image tests (default tests/visual/scenarios.json), and a page reviewing their shots",
    },
    CommandSpec { name: "bench", operands: "", checkout: true, about: "guest time per frame, without rendering" },
    CommandSpec {
        name: "packets",
        operands: "DIR",
        checkout: true,
        about: "each frame's input and render packet, without rendering",
    },
    CommandSpec {
        name: "test-guest",
        operands: "[FILE...]",
        checkout: true,
        about: "the checkout's guest tests, compiled and laid out by its compiler plugin (pen.json)",
    },
    CommandSpec {
        name: "test-blot",
        operands: "[FILE...]",
        checkout: true,
        about: "alias for test-guest (Blot checkouts: tests/*.blot, tests/traps/ and tests/rejected/)",
    },
    CommandSpec {
        name: "gltf",
        operands: "FILE",
        checkout: false,
        about: "import a glTF file (.gltf or .glb) as assets: a mesh JSON for each primitive (skinned ones with \
                their rig and the file's animations as clips), its materials and their images, and \
                NAME.import.json listing them; an asset recipe's work",
    },
    CommandSpec {
        name: "import-scene",
        operands: "FILE",
        checkout: false,
        about: "import a glTF 2.0 scene (.gltf or .glb) into unpacked assets plus a neutral .scene.json \
                manifest preserving node trees, cameras, punctual lights, -col/-colonly/-convcol colliders, \
                _LOD0..3 groups, and a re-import preset",
    },
    CommandSpec {
        name: "bake-lightmaps",
        operands: "FILE",
        checkout: false,
        about: "bake multi-bounce GPU path-traced lightmaps (HDR irradiance EXR, directional map, sun shadowmask) \
                with automatic UV2 chart unwrapping and gutter dilation from a scene bake job JSON",
    },
    CommandSpec {
        name: "diverge",
        operands: "RECORDING",
        checkout: true,
        about: "play an input recording (.penrec) on this build and on another (--against), without rendering, \
                and say where they first differ: the frame, its packet's fields and the state's lanes, by saved \
                section; fails if they differ (for `git bisect run`)",
    },
    CommandSpec {
        name: "fuzz",
        operands: "",
        checkout: true,
        about: "random input against the game's invariants, without rendering: each run plays steps a seed \
                makes up (keys, chords, clicks, drags, the wheel, console commands, tunables, actions on its \
                interface's nodes) and checks that the game does not trap, its packets are valid and its state \
                finite; a run that breaks one is cut down to the steps that still do, written as a script \
                (--script-file plays one back with the same checks)",
    },
    CommandSpec {
        name: "report",
        operands: "FILE",
        checkout: false,
        about: "what a bug report (.penbug: Ctrl+Shift+B in a window, or a script's report step) holds: its \
                build, system and GPU, the end of its log and its recording of the last 30 s, which `--replay \
                FILE` plays; --out DIR unpacks it",
    },
    CommandSpec {
        name: "new",
        operands: "DIR",
        checkout: false,
        about: "a game to start from in the new directory DIR, named after it: pen.json, a Blot program that \
                turns a cube, its assets, a test and pen's generated bindings; it compiles and runs as it is",
    },
    CommandSpec {
        name: "texture",
        operands: "IMAGE",
        checkout: false,
        about: "a PNG as KTX2 files a GPU samples directly, one per format family (name.bc7.ktx2, \
                name.astc.ktx2, name.etc2.ktx2, name.rgba.ktx2), every mip level worked out; a material \
                names name.ktx2 and the renderer reads the family its device has",
    },
    CommandSpec {
        name: "font",
        operands: "FONT",
        checkout: false,
        about: "a TrueType or OpenType font's multi-channel signed distance field atlas (KTX2, RGBA8 read as \
                data) and its glyphs' metrics in ems (the same name's .json), for text sharp at any size",
    },
    CommandSpec {
        name: "patch",
        operands: "OLD NEW",
        checkout: false,
        about: "a patch from one package version to another (--out FILE): package folders, or versions in the \
                store pen build keeps them in (a version's digest or its start; --store, by default the \
                checkout's .pen/store)",
    },
    CommandSpec {
        name: "apply",
        operands: "PATCH [DIR]",
        checkout: false,
        about: "apply a patch to the package in DIR (by default the one beside this executable): only to the \
                version it was made from, every file checked, the new version built beside the old one and \
                then swapped in",
    },
    CommandSpec {
        name: "loudness",
        operands: "FILE [FILE...]",
        checkout: false,
        about: "each WAV or Ogg Opus clip's loudness (EBU R128, LUFS), true peak (dBTP) and length; with \
                --normalize, each brought to its target loudness as far as its true peak allows: a WAV's \
                samples scaled, an Opus file's output gain set",
    },
    CommandSpec {
        name: "clean",
        operands: "",
        checkout: false,
        about: "delete pen's caches in the platform cache directory: compiled games, plugins and GPU \
                pipelines, and what plugins run and keep",
    },
    CommandSpec {
        name: "serve",
        operands: "",
        checkout: false,
        about: "receive packages that `pen push` sends (on a Steam Deck, say), each into a directory of its \
                name, only the files that changed; it listens on 127.0.0.1:7811 unless --listen says where, \
                and takes pushes with its token only",
    },
    CommandSpec {
        name: "relay",
        operands: "",
        checkout: false,
        about: "run a multi-room NAT-traversal netplay relay accepting both native TCP and browser WebSocket \
                peers on the same port (default 0.0.0.0:7813 unless --listen says where)",
    },
    CommandSpec {
        name: "push",
        operands: "ADDRESS",
        checkout: false,
        about: "send a package (default .pen/package) to `pen serve` at ADDRESS (port 7811 unless given): \
                only the files the server lacks, checked by their SHA-256",
    },
    CommandSpec {
        name: "remote",
        operands: "ADDRESS [REQUEST...]",
        checkout: false,
        about: "ask a game run with --devtools-listen (port 7812 unless given) one dev tools request, an \
                operation of protocol/schema.json's DevtoolsOperation and up to four numbers (default \
                frame), and print the answer",
    },
    CommandSpec {
        name: "check-workers",
        operands: "DIR",
        checkout: false,
        about: "verify packaged native workers (game.cwasm and optional workers.cwasm) without a Wasm compiler",
    },
];

#[cfg(not(feature = "compiler"))]
const COMMANDS: &[CommandSpec] = &[
    CommandSpec { name: "apply", operands: "PATCH [DIR]", about: "apply a package update" },
    CommandSpec {
        name: "check-workers",
        operands: "DIR",
        about: "verify packaged native workers (game.cwasm and optional workers.cwasm)",
    },
];

/// Playing: a packaged game, `dev` and `run`.
const PLAY: &[&str] = &["", "dev", "run"];
/// Every command that works on a checkout.
const CHECKOUT: &[&str] = &[
    "tool",
    "dev",
    "web",
    "run",
    "build",
    "check",
    "assets",
    "codegen",
    "export",
    "test",
    "bench",
    "packets",
    "test-guest",
    "test-blot",
    "diverge",
    "fuzz",
];

const OPTIONS: &[OptionSpec] = &[
    OptionSpec { name: "--help", value: "", commands: &[], place: Place::Prose, about: "this" },
    OptionSpec {
        name: "--out",
        value: "FILE",
        commands: &["texture", "font"],
        place: Place::Command,
        about: "the .ktx2 key to write (default IMAGE's or FONT's name with .ktx2)",
    },
    OptionSpec {
        name: "--linear",
        value: "",
        commands: &["texture"],
        place: Place::Command,
        about: "its texels are data (normals, masks), not sRGB colour",
    },
    OptionSpec {
        name: "--flat",
        value: "",
        commands: &["texture"],
        place: Place::Command,
        about: "the full size alone, no smaller mip levels",
    },
    OptionSpec {
        name: "--em",
        value: "N",
        commands: &["font"],
        place: Place::Command,
        about: "pixels per em the glyphs are drawn at (default 48)",
    },
    OptionSpec {
        name: "--range",
        value: "N",
        commands: &["font"],
        place: Place::Command,
        about: "pixels of distance either side of an edge the field holds (default 4)",
    },
    OptionSpec {
        name: "--width",
        value: "N",
        commands: &["font"],
        place: Place::Command,
        about: "the atlas's width in pixels; its height is the power of two that fits (default 512)",
    },
    OptionSpec {
        name: "--weight",
        value: "N",
        commands: &["font"],
        place: Place::Command,
        about: "a variable font's weight (its wght axis)",
    },
    OptionSpec {
        name: "--chars",
        value: "TEXT",
        commands: &["font"],
        place: Place::Command,
        about: "the characters to draw (default the printable ASCII ones)",
    },
    OptionSpec { name: "--version", value: "", commands: &[], place: Place::Prose, about: "pen's version" },
    OptionSpec { name: "--project", value: "DIR", commands: CHECKOUT, place: Place::Prose, about: "the checkout" },
    OptionSpec {
        name: "--no-pipeline",
        value: "",
        commands: &["dev"],
        place: Place::Command,
        about: "leave the asset recipes be",
    },
    OptionSpec {
        name: "--port",
        value: "N",
        commands: &["web", "test"],
        place: Place::Command,
        about: "listen on port N (default the first free one from 8000 to 8015)",
    },
    OptionSpec {
        name: "--out",
        value: "DIR",
        commands: &["build"],
        place: Place::Command,
        about: "where (default .pen/package, or .pen/web-export with --web)",
    },
    OptionSpec {
        name: "--runtime",
        value: "EXE",
        commands: &["build"],
        place: Place::Command,
        about: "the runtime to ship (default player/ beside this executable); it must be \
                built from the same pen",
    },
    OptionSpec {
        name: "--no-optimize",
        value: "",
        commands: &["build"],
        place: Place::Command,
        about: "keep the compiler's Wasm unchanged (build normally runs wasm-opt -O3 before shipping)",
    },
    OptionSpec {
        name: "--target",
        value: "TRIPLE",
        commands: &["build"],
        place: Place::Command,
        about: "the runtime's Rust target, when its executable header does not say",
    },
    OptionSpec {
        name: "--cpu",
        value: "PRESET",
        commands: &["build"],
        place: Place::Command,
        about: "the CPU game.cwasm is compiled for: a Cranelift preset, or native (default x86-64-v2 on x86-64)",
    },
    OptionSpec {
        name: "--bundle-id",
        value: "ID",
        commands: &["build"],
        place: Place::Command,
        about: "a macOS runtime's application bundle identifier (default pen.NAME)",
    },
    OptionSpec {
        name: "--web",
        value: "",
        commands: &["build"],
        place: Place::Command,
        about: "the Rust browser host and the game as static files instead",
    },
    OptionSpec {
        name: "--web-compact",
        value: "",
        commands: &["build"],
        place: Place::Command,
        about: "with --web, build a tailored single-player host containing only the features this game uses",
    },
    OptionSpec {
        name: "--web-features",
        value: "LIST",
        commands: &["build"],
        place: Place::Command,
        about: "with --web, pass an explicit comma-separated pen-web feature list (empty for bare core)",
    },
    OptionSpec {
        name: "--worker-deadlines",
        value: "",
        commands: &["build"],
        place: Place::Command,
        about: "also precompile epoch-interrupted workers.cwasm and workers.meta for deadline-bounded worker pools",
    },
    OptionSpec {
        name: "--steam",
        value: "",
        commands: &["build"],
        place: Place::Command,
        about: "also write steamcmd's app and depot build scripts for the package in steam/ beside it, from \
                pen.json's \"steam\" section ({\"app\": ID, \"depots\": {\"linux\": ID, \"windows\": ID}, \
                \"branch\": NAME}), and say the steamcmd run that uploads them",
    },
    OptionSpec {
        name: "--itch",
        value: "",
        commands: &["build"],
        place: Place::Command,
        about: "also say the butler push of the package to itch.io, from pen.json's \"itch\" section \
                ({\"target\": \"user/game\"}): to its platform's channel, as a patch",
    },
    OptionSpec {
        name: "--upload",
        value: "",
        commands: &["build"],
        place: Place::Command,
        about: "run those uploads: steamcmd as STEAM_BUILD_USER (logged in before), butler with \
                BUTLER_API_KEY; without it, nothing is uploaded",
    },
    OptionSpec {
        name: "--label",
        value: "TEXT",
        commands: &["build"],
        place: Place::Command,
        about: "the build's description on Steam and version on itch.io",
    },
    OptionSpec {
        name: "--zip",
        value: "FILE",
        commands: &["build"],
        place: Place::Command,
        about: "with --web, also the site as a ZIP to upload (itch.io's limits are checked)",
    },
    OptionSpec {
        name: "--check",
        value: "",
        commands: &["codegen"],
        place: Place::Command,
        about: "check the generated code instead of writing it",
    },
    OptionSpec {
        name: "--check",
        value: "",
        commands: &["assets"],
        place: Place::Command,
        about: "build nothing: say which steps' outputs differ from what the cache built from their sources, and \
                fail if any do",
    },
    OptionSpec {
        name: "--only",
        value: "STEP",
        commands: &["assets"],
        place: Place::Command,
        about: "only this step (again for more)",
    },
    OptionSpec {
        name: "--cache",
        value: "DIR",
        commands: &["assets"],
        place: Place::Command,
        about: "the asset cache (default PEN_ASSET_CACHE, else assets/ in pen's cache directory)",
    },
    OptionSpec {
        name: "--rust",
        value: "FILE",
        commands: &["codegen"],
        place: Place::Command,
        about: "where the Rust protocol code goes",
    },
    OptionSpec { name: "--out", value: "DIR", commands: &["export"], place: Place::Command, about: "where (required)" },
    OptionSpec { name: "--out", value: "FILE", commands: &["patch"], place: Place::Command, about: "where (required)" },
    OptionSpec {
        name: "--store",
        value: "DIR",
        commands: &["build", "patch"],
        place: Place::Command,
        about: "the store of package versions (default .pen/store in the checkout)",
    },
    OptionSpec {
        name: "--out",
        value: "DIR",
        commands: &["gltf", "import-scene"],
        place: Place::Command,
        about: "the asset directory written into (default src/assets)",
    },
    OptionSpec {
        name: "--out",
        value: "DIR",
        commands: &["bake-lightmaps"],
        place: Place::Command,
        about: "output directory for baked lightmap images and manifest (default lightmaps/)",
    },
    OptionSpec {
        name: "--size",
        value: "N",
        commands: &["bake-lightmaps"],
        place: Place::Command,
        about: "override square lightmap atlas resolution in texels (default from job JSON or 256)",
    },
    OptionSpec {
        name: "--samples",
        value: "N",
        commands: &["bake-lightmaps"],
        place: Place::Command,
        about: "override path-traced samples per texel (1..=256)",
    },
    OptionSpec {
        name: "--bounces",
        value: "N",
        commands: &["bake-lightmaps"],
        place: Place::Command,
        about: "override indirect diffuse bounce count (1..=4)",
    },
    OptionSpec {
        name: "--name",
        value: "NAME",
        commands: &["gltf", "import-scene"],
        place: Place::Command,
        about: "what the assets' names start with (default the file's name)",
    },
    OptionSpec {
        name: "--shader",
        value: "KEY",
        commands: &["gltf", "import-scene"],
        place: Place::Command,
        about: "the shader the materials name (default scene.wgsl)",
    },
    OptionSpec {
        name: "--scene",
        value: "FILE",
        commands: &["import-scene"],
        place: Place::Command,
        about: "explicit output path for the generated .scene.json manifest (default <out>/<name>.scene.json)",
    },
    OptionSpec {
        name: "--scale",
        value: "FACTOR",
        commands: &["import-scene"],
        place: Place::Command,
        about: "uniform root scale applied to imported scene root nodes (default 1.0)",
    },
    OptionSpec {
        name: "--vat",
        value: "FPS",
        commands: &["gltf"],
        place: Place::Command,
        about: "also bake each skinned primitive's clips into a vertex-animation texture, FPS frames a second, \
                with a material drawing with it",
    },
    OptionSpec {
        name: "--vat-shader",
        value: "KEY",
        commands: &["gltf"],
        place: Place::Command,
        about: "the shader those materials name (default vat.wgsl)",
    },
    OptionSpec {
        name: "--blot",
        value: "DIR",
        commands: &["new"],
        place: Place::Command,
        about: "the Blot checkout (blot2) it builds with (default PEN_BLOT, else a blot2 beside DIR or a \
                directory above it)",
    },
    OptionSpec {
        name: "--update",
        value: "",
        commands: &["test", "test-guest", "test-blot"],
        place: Place::Command,
        about: "write the shots as the golden images",
    },
    OptionSpec {
        name: "--review",
        value: "",
        commands: &["test"],
        place: Place::Command,
        about: "serve the review page every run writes (OUT/review.html: each shot beside its golden image and \
                their difference), with buttons that accept a shot as its golden image, until stopped",
    },
    OptionSpec {
        name: "--last",
        value: "",
        commands: &["test"],
        place: Place::Command,
        about: "with --review, serve the last run's page instead of running (no GPU)",
    },
    OptionSpec {
        name: "--only",
        value: "NAME",
        commands: &["test"],
        place: Place::Command,
        about: "only the scenarios whose names hold NAME (again for more)",
    },
    OptionSpec {
        name: "--golden",
        value: "DIR",
        commands: &["test", "test-guest", "test-blot"],
        place: Place::Command,
        about: "the golden images (default golden/ beside FILE; tests/scenarios/golden for --session)",
    },
    OptionSpec {
        name: "--out",
        value: "DIR",
        commands: &["test", "test-guest", "test-blot"],
        place: Place::Command,
        about: "where shots and differences go (default .pen/visual; .pen/session-tests for --session)",
    },
    OptionSpec {
        name: "--session",
        value: "",
        commands: &["test-guest", "test-blot"],
        place: Place::Command,
        about: "run guest driver tests against isolated native headless sessions, sequentially (explicit test files required)",
    },
    OptionSpec {
        name: "--filter",
        value: "TEXT",
        commands: &["test-guest", "test-blot"],
        place: Place::Command,
        about: "only the tests whose names hold TEXT",
    },
    OptionSpec {
        name: "--jobs",
        value: "N",
        commands: &["test-guest", "test-blot"],
        place: Place::Command,
        about: "modules compiled at once, each worker in one compiler session, and tests run at once (default half \
                the cores, at most 8)",
    },
    OptionSpec {
        name: "--runs",
        value: "N",
        commands: &["bench"],
        place: Place::Command,
        about: "run the frames N times, keeping each metric's best (default 1)",
    },
    OptionSpec {
        name: "--json",
        value: "",
        commands: &["bench"],
        place: Place::Command,
        about: "print the results on stdout as JSON, for --compare",
    },
    OptionSpec {
        name: "--compare",
        value: "FILE",
        commands: &["bench"],
        place: Place::Command,
        about: "compare with another build's --json results (again for the best of several), and fail when a \
                metric is slower than its threshold",
    },
    OptionSpec {
        name: "--threshold",
        value: "PCT",
        commands: &["bench"],
        place: Place::Command,
        about: "how much slower every metric may get, in percent (default each its own: the median and mean 10, \
                p95 and snapshots 20, loading 25)",
    },
    OptionSpec {
        name: "--against",
        value: "DIR",
        commands: &["diverge"],
        place: Place::Command,
        about: "the other build: a game bundle (`pen export`'s, or a package's game/), as of the recording's \
                commit",
    },
    OptionSpec {
        name: "--seed",
        value: "N",
        commands: &["fuzz"],
        place: Place::Command,
        about: "the first run's seed; each run after takes the next (default 1)",
    },
    OptionSpec {
        name: "--runs",
        value: "N",
        commands: &["fuzz"],
        place: Place::Command,
        about: "how many (default 10)",
    },
    OptionSpec {
        name: "--out",
        value: "DIR",
        commands: &["fuzz"],
        place: Place::Command,
        about: "where the scripts of runs that break the game go (default .pen/fuzz)",
    },
    OptionSpec {
        name: "--out",
        value: "DIR",
        commands: &["report"],
        place: Place::Command,
        about: "unpack its files there (report.txt, log.txt, recording.penrec, state.bin)",
    },
    OptionSpec {
        name: "--deadline",
        value: "SECONDS",
        commands: &["test-guest", "test-blot"],
        place: Place::Command,
        about: "how long one test may run before it fails as stuck (default 600; the time it waits for a core \
                counts)",
    },
    OptionSpec {
        name: "--normalize",
        value: "",
        commands: &["loudness"],
        place: Place::Command,
        about: "bring each clip to its target loudness, rewriting it",
    },
    OptionSpec {
        name: "--lufs",
        value: "N",
        commands: &["loudness"],
        place: Place::Command,
        about: "the target loudness (default -23, EBU R128's, or the targets file's)",
    },
    OptionSpec {
        name: "--ceiling",
        value: "N",
        commands: &["loudness"],
        place: Place::Command,
        about: "the highest true peak, in dBTP, normalizing may take a clip to (default -1)",
    },
    OptionSpec {
        name: "--targets",
        value: "FILE",
        commands: &["loudness"],
        place: Place::Command,
        about: "targets as JSON: {\"target\": LUFS, \"peak\": dBTP, \"sounds\": {\"NAME\": LUFS}}",
    },
    OptionSpec { name: "--mute", value: "", commands: PLAY, place: Place::Play, about: "no sound" },
    OptionSpec {
        name: "--audio-buffer",
        value: "N",
        commands: PLAY,
        place: Place::Play,
        about: "the audio device's buffer, N samples within what it takes (default the system's; smaller plays \
                sooner, at more risk of gaps)",
    },
    OptionSpec {
        name: "--headless",
        value: "",
        commands: PLAY,
        place: Place::Play,
        about: "no window; draw offscreen (steady clock, 60 fps unless --fps)",
    },
    OptionSpec {
        name: "--size",
        value: "WxH",
        commands: PLAY,
        place: Place::Play,
        about: "the window's or the offscreen image's size in pixels (headless default 1920x1280, where \
                script coordinates match the browser's)",
    },
    OptionSpec {
        name: "--fps",
        value: "N",
        commands: PLAY,
        place: Place::Play,
        about: "steady game clock: N frames per game second",
    },
    OptionSpec {
        name: "--frames",
        value: "N",
        commands: &["", "dev", "run", "bench", "packets", "fuzz"],
        place: Place::Play,
        about: "stop after N frames (bench: default 600; packets: default 120, or as long as the script lasts; \
                fuzz: each run's, default 600)",
    },
    OptionSpec {
        name: "--screenshot",
        value: "PNG",
        commands: PLAY,
        place: Place::Play,
        about: "write the last frame",
    },
    OptionSpec {
        name: "--shots",
        value: "DIR",
        commands: PLAY,
        place: Place::Play,
        about: "where shot=NAME steps write NAME.png (default shots)",
    },
    OptionSpec {
        name: "--record",
        value: "VIDEO",
        commands: PLAY,
        place: Place::Play,
        about: "record every frame (.mp4, .mkv or .webm, with sound unless --mute)",
    },
    OptionSpec {
        name: "--script",
        value: "STEPS",
        commands: &["", "dev", "run", "bench", "packets", "fuzz"],
        place: Place::Play,
        about: "scripted input: wait=N key=K down=K up=K type=T move=X,Y glide=X,Y,FRAMES click=X,Y press=B release=B \
                drag=X,Y,X,Y wheel=D[,X,Y] shot=NAME save=F load=F quit; ui=ID,ACTION[,VALUE] (an assistive \
                technology's action on an interface node: focus, blur, activate, increment, decrement, \
                set_value, expand or collapse); tune=NAME=VALUE (a live-tunable constant of the checkout's \
                src/tunables.json); overlay=/COMMAND[,ARG...] (a line typed into the dev tools' console); \
                pad=connect|disconnect|BUTTON[,VALUE]|AXIS,VALUE (a gamepad); touch=ID,X,Y lift=ID (a touch \
                screen); motion=DX,DY (the locked mouse); compose=TEXT[,CURSOR] commit=TEXT (an input method); \
                overlay-shot=NAME (a shot with the dev tools over it); \
                glide eases from the last scripted pointer position (initially 0,0) over positive FRAMES, \
                then continues; move is instant",
    },
    OptionSpec {
        name: "--script-file",
        value: "FILE",
        commands: &["", "dev", "run", "bench", "packets", "fuzz"],
        place: Place::Play,
        about: "the same, from a file (lines starting with # are comments)",
    },
    OptionSpec {
        name: "--data",
        value: "DIR",
        commands: PLAY,
        place: Place::Play,
        about: "keep saves, preferences and crash reports in DIR instead of the game's data directory",
    },
    OptionSpec {
        name: "--settle",
        value: "",
        commands: PLAY,
        place: Place::Play,
        about: "draw the first frame until every asset has loaded before the clock starts (default for \
                --headless and --record)",
    },
    OptionSpec { name: "--no-settle", value: "", commands: PLAY, place: Place::Play, about: "start the clock at once" },
    OptionSpec {
        name: "--mod",
        value: "DIR",
        commands: &["", "dev", "run", "packets"],
        place: Place::Play,
        about: "a mod beside the game, again for more: DIR's mod.json and mod.wasm, run in a sandbox with the \
                capabilities its manifest asks for (docs/mods.md)",
    },
    OptionSpec {
        name: "--game",
        value: "DIR",
        commands: &["", "build", "test", "bench", "packets", "diverge", "fuzz"],
        place: Place::Play,
        about: "a game bundle (`pen export`'s, a package's game/, or one built in another language) instead of \
                the checkout's",
    },
    OptionSpec {
        name: "--assets",
        value: "DIR",
        commands: &["", "test", "bench", "packets", "diverge", "fuzz"],
        place: Place::Play,
        about: "the bundle's assets from elsewhere",
    },
    OptionSpec {
        name: "--scene",
        value: "FILE",
        commands: &["", "test", "bench", "packets", "diverge", "fuzz"],
        place: Place::Play,
        about: "the bundle's scene from elsewhere, which the editor saves to",
    },
    OptionSpec {
        name: "--frame-budget",
        value: "MS",
        commands: PLAY,
        place: Place::Play,
        about: "dynamic resolution: lower the scene's render scale (to half at most) while the GPU takes \
                longer than MS milliseconds a frame, and raise it again when it takes well under",
    },
    OptionSpec {
        name: "--software",
        value: "",
        commands: &["", "dev", "run", "test", "test-guest", "test-blot"],
        place: Place::Play,
        about: "draw on a software GPU adapter (lavapipe, WARP), as PEN_SOFTWARE=1 does",
    },
    OptionSpec {
        name: "--record-input",
        value: "FILE",
        commands: PLAY,
        place: Place::Play,
        about: "record the session's input (a .penrec) to replay it exactly",
    },
    OptionSpec {
        name: "--replay",
        value: "FILE",
        commands: PLAY,
        place: Place::Play,
        about: "play recorded input instead of the player's: the same frames, bit for bit (headless, the run \
                ends with it; in a window, it pauses there)",
    },
    OptionSpec {
        name: "--history",
        value: "",
        commands: PLAY,
        place: Place::Play,
        about: "keep the last 30 s to rewind, replay and report (always in a window, dev and replays)",
    },
    OptionSpec {
        name: "--overlap",
        value: "",
        commands: PLAY,
        place: Place::Play,
        about: "compute the next frame while this one draws (one frame more input latency)",
    },
    OptionSpec {
        name: "--trace",
        value: "FILE",
        commands: PLAY,
        place: Place::Play,
        about: "a line per frame: its number, and its input's and packet's hashes",
    },
    OptionSpec {
        name: "--broadcast",
        value: "ADDRESS",
        commands: PLAY,
        place: Place::Play,
        about: "stream the final frames to viewers on ADDRESS (HOST:PORT), over WebSockets",
    },
    OptionSpec {
        name: "--spectate",
        value: "ADDRESS",
        commands: PLAY,
        place: Place::Play,
        about: "watch the stream at ADDRESS (HOST:PORT or ws://...) as it comes, as a replay plays",
    },
    OptionSpec {
        name: "--net",
        value: "PLAYER:ADDRESS,...",
        commands: PLAY,
        place: Place::Play,
        about: "play as PLAYER (from 0) of a net session, every player's UDP address in order (docs/netplay.md)",
    },
    OptionSpec {
        name: "--overlay",
        value: "",
        commands: &["dev", "run"],
        place: Place::Play,
        about: "open the dev tools' console from the start (or PEN_OVERLAY=1)",
    },
    OptionSpec {
        name: "--overlay-shot",
        value: "PNG",
        commands: &["dev", "run"],
        place: Place::Play,
        about: "write the last frame with the dev tools over it (headless too)",
    },
    OptionSpec {
        name: "--devtools-listen",
        value: "ADDRESS",
        commands: PLAY,
        place: Place::Play,
        about: "answer remote dev tools (`pen remote`) at ADDRESS, such as 127.0.0.1:7812",
    },
    OptionSpec {
        name: "--listen",
        value: "ADDRESS",
        commands: &["serve", "relay"],
        place: Place::Command,
        about: "where to listen (serve default 127.0.0.1:7811; relay default 0.0.0.0:7813)",
    },
    OptionSpec {
        name: "--dir",
        value: "DIR",
        commands: &["serve"],
        place: Place::Command,
        about: "where packages go (default the data directory's pushed/)",
    },
    OptionSpec {
        name: "--token",
        value: "TEXT",
        commands: &["serve", "push"],
        place: Place::Command,
        about: "the server's token, which a push must give (or PEN_PUSH_TOKEN; serve makes one up and says it)",
    },
    OptionSpec {
        name: "--allow-run",
        value: "",
        commands: &["serve"],
        place: Place::Command,
        about: "start what a push asks to (after stopping what it started before)",
    },
    OptionSpec {
        name: "--package",
        value: "DIR",
        commands: &["push"],
        place: Place::Command,
        about: "the package to send (default .pen/package)",
    },
    OptionSpec {
        name: "--name",
        value: "NAME",
        commands: &["push"],
        place: Place::Command,
        about: "its name on the server (default the game's, from the package's game/game.json)",
    },
    OptionSpec {
        name: "--run",
        value: "FILE",
        commands: &["push"],
        place: Place::Command,
        about: "start this file of the package there once it is in place (the server must --allow-run)",
    },
    OptionSpec {
        name: "--text",
        value: "TEXT",
        commands: &["remote"],
        place: Place::Command,
        about: "the request's text (a file's path, a preset's name)",
    },
    OptionSpec {
        name: "--follow",
        value: "",
        commands: &["remote"],
        place: Place::Command,
        about: "with log: print the log as it grows",
    },
];

/// Every variable of the environment pen reads, and what it does.
#[cfg(feature = "compiler")]
const ENVIRONMENT: [(&str, &str); 38] = [
    ("PEN_PRESENT_MODE=MODE", "auto-vsync (the default), auto-no-vsync, fifo, fifo-relaxed, immediate or mailbox"),
    ("PEN_SOFTWARE=1", "as --software"),
    ("PEN_HDR=1", "HDR output where the display and surface can (not while recording or taking shots)"),
    ("PEN_OVERLAY=1", "as --overlay"),
    (
        "PEN_TRACE_SYSTEMS=FILE",
        "capture per-system CPU time and allocation traffic to Chrome/Perfetto JSON and a .summary.json (native development runtime)",
    ),
    ("PEN_TRACE_LIMIT=N", "maximum timeline events in a system trace (default 1000000); summaries keep counting"),
    ("PEN_SMOKE_FRAMES=N", "as --frames N"),
    ("PEN_AUDIO_BUFFER=N", "as --audio-buffer N"),
    (
        "PEN_PROFILE_FRAMES=N",
        "after PEN_PROFILE_WARMUP=N frames (default 90), time N frames' phases, say each one's median and stop",
    ),
    (
        "PEN_FRAME_TIMES=1",
        "a line on stderr for each frame, with its phases' times, and the renderer's pipeline compile times",
    ),
    (
        "RUST_LOG=LEVEL",
        "more of the libraries' log on stderr: debug, or per crate (pen_render=debug); warnings always show",
    ),
    (
        "PEN_STARTUP=1",
        "when each phase of starting is reached: the program loaded, the GPU ready, the game's first frame, the \
         first frame drawn whole and settling",
    ),
    ("PEN_GPU_BACKTRACE=1", "where a GPU validation error came from"),
    ("PEN_SERVICES_LOG=1", "a line for each request the game makes of its services"),
    (
        "PEN_A11Y_DUMP=FILE",
        "the game's interface as assistive technology sees it, written to FILE whenever it changes (headless too)",
    ),
    ("PEN_A11Y_ATSPI=1", "headless, the game's interface on the AT-SPI bus, at the game's pace (Linux; tests/atspi)"),
    (
        "PEN_A11Y_AUDIT=FILE",
        "the accessibility audit of the game's interface and its pixels, written to FILE whenever it changes",
    ),
    ("PEN_LOCALE=TAG", "the language the game's string tables show in (default the system's locale)"),
    (
        "PEN_COLOR_FILTER=NAME[:STRENGTH]",
        "a colour filter for this run: protanopia, deuteranopia, tritanopia or achromatopsia; simulate-NAME shows what a player with it sees",
    ),
    ("PEN_TEXT_SCALE=N", "the engine's text scale for this run, 1 to 2"),
    ("PEN_REDUCED_MOTION=1", "the engine's reduced motion for this run"),
    ("PEN_RECORD_PCM=FILE", "a recording's sound as mixed, before it is encoded (f32 stereo at 48 kHz)"),
    ("PEN_STEAM=1", "Steam outside a package (pen built with Steam)"),
    ("PEN_STEAM_DEBUG=STEPS", "achievement and stat steps to run against Steam (see steam.rs)"),
    ("PEN_STEAM_APP_ID=ID", "relaunch through Steam when started outside it"),
    ("STEAM_BUILD_USER=NAME", "build --steam --upload: the Steamworks account steamcmd logged in as before"),
    ("PEN_STEAMCMD=EXE, PEN_BUTLER=EXE", "build --upload: steamcmd and butler, if not on PATH"),
    ("BUTLER_API_KEY=KEY", "build --itch --upload: butler's itch.io API key"),
    (
        "PEN_GRAPHICS_TIER=TIER",
        "the graphics preset for a player without stored preferences: auto (from the GPU, and in headless runs \
         too), software, handheld, low, medium or high",
    ),
    ("PEN_CRASH_TEST=N", "panic at frame N, to try crash reports"),
    ("PEN_REPORTS=off", "send no crash or performance reports, whatever the player agreed to"),
    ("PEN_PLUGINS=DIR", "where pen's own plugins are (default plugins/ beside pen)"),
    (
        "PEN_ASSET_CACHE=DIR",
        "what asset steps built, by their sources (default assets/ in pen's cache directory); share one between \
         checkouts and CI runs",
    ),
    (
        "PEN_WEB_HOST=DIR",
        "a built browser host, for web and build --web (default web/ beside pen, else built from this repository)",
    ),
    (
        "BLOT_INLINING=0, =1",
        "Cranelift's inlining: =0 turns it off where it is on (run, build, a package), =1 on where it is off \
         (dev and the dev tools)",
    ),
    ("PEN_BUILD=TEXT", "while building pen: a label for its crash reports"),
    ("PEN_PUSH_TOKEN=TEXT", "as --token, for pen serve and pen push"),
    (
        "SOURCE_DATE_EPOCH=SECONDS",
        "build: every file and directory of the package takes this modification time, so that an archive \
         of it is reproducible too",
    ),
];

/// A line of the usage: `head`, then `about` from column 22, wrapped at 100.
fn entry(text: &mut String, head: &str, about: &str) {
    const COLUMN: usize = 22;
    const WIDTH: usize = 100;
    let mut line = head.to_owned();
    if line.chars().count() >= COLUMN - 1 {
        text.push_str(&line);
        text.push('\n');
        line.clear();
    }
    for word in about.split_whitespace() {
        let width = line.chars().count();
        if width > COLUMN && width + 1 + word.chars().count() > WIDTH {
            text.push_str(&line);
            text.push('\n');
            line.clear();
        }
        let width = line.chars().count();
        line.push_str(&" ".repeat(COLUMN.saturating_sub(width).max(1)));
        line.push_str(word);
    }
    text.push_str(&line);
    text.push('\n');
}

/// `pen` for no command, else `pen COMMAND`.
fn invocation(command: &str) -> String {
    if command.is_empty() { "pen".into() } else { format!("pen {command}") }
}

/// Names in prose: `a`, `a and b`, `a, b and c`.
fn listed(names: &[String]) -> String {
    match names {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

#[cfg(feature = "compiler")]
pub(crate) fn usage() -> String {
    let mut text = String::from(
        "usage: pen [COMMAND] [OPTIONS]  (pen --help: this; pen --version)\n\n\
         without a command, pen plays a packaged game: game/ beside the executable (a `pen build` package),\n\
         or --game DIR.\n\n\
         commands, in a checkout (a directory with pen.json and src/main.blot; or --project DIR):\n",
    );
    let command = |text: &mut String, spec: &CommandSpec| {
        let head = format!("  {} {}", spec.name, spec.operands);
        entry(text, head.trim_end(), spec.about);
        for option in
            OPTIONS.iter().filter(|option| option.place == Place::Command && option.commands.contains(&spec.name))
        {
            entry(text, format!("    {} {}", option.name, option.value).trim_end(), option.about);
        }
    };
    for spec in COMMANDS.iter().filter(|spec| spec.checkout) {
        command(&mut text, spec);
    }
    text.push_str("\nand anywhere:\n");
    for spec in COMMANDS.iter().filter(|spec| !spec.checkout) {
        command(&mut text, spec);
    }
    text.push_str("\noptions for playing (pen, dev and run):\n");
    for option in OPTIONS.iter().filter(|option| option.place == Place::Play) {
        let named =
            |commands: Vec<&&str>| listed(&commands.into_iter().map(|command| invocation(command)).collect::<Vec<_>>());
        let extra: Vec<&&str> = option.commands.iter().filter(|command| !PLAY.contains(command)).collect();
        let missing = PLAY.iter().any(|command| !option.commands.contains(command));
        let about = match (extra.is_empty(), missing) {
            (true, false) => option.about.to_owned(),
            (false, false) => format!("{}; also {}", option.about, named(extra)),
            (true, true) => format!("{}; {} only", option.about, named(option.commands.iter().collect())),
            (false, true) => format!("{}; for {}", option.about, named(option.commands.iter().collect())),
        };
        entry(&mut text, format!("  {} {}", option.name, option.value).trim_end(), &about);
    }
    text.push_str(
        "\ntime travel: Ctrl+Shift+P pauses and plays, Ctrl+Shift+period steps a frame, Ctrl+Shift+comma\n\
         steps back, Ctrl+Shift+R replays the last 5 s (after a reload, on the new build); in scripts,\n\
         pause resume step[=N] rewind=N replay=SECONDS.\n\n\
         bug reports: Ctrl+Shift+B in a window (a script's report[=FILE]) writes the last 30 s of input,\n\
         the state, the log and the system to a .penbug in reports/ of the data directory (pen report).\n\n\
         dev tools (dev and run): the checkout's own program for them (devtools/main.blot) runs beside the\n\
         game and draws over it: its console on the backquote key (/help lists what it does), and its\n\
         diagnostics; a checkout without one runs without dev tools.\n\n\
         environment:\n",
    );
    for (name, about) in ENVIRONMENT {
        entry(&mut text, &format!("  {name}"), about);
    }
    text.push_str(
        "\nthe game's name (its data directory, and its executable in a package) and title come from the\n\
         checkout's pen.json, {\"game\": {\"name\": \"...\", \"title\": \"...\"}}, else its directory's name; a\n\
         package keeps them in game/game.json. A panic, or an error that stops a package's runtime, writes\n\
         a crash report to crashes/ in the data directory (or --data DIR).",
    );
    text
}

#[cfg(not(feature = "compiler"))]
pub(crate) fn usage() -> String {
    let mut text = String::from(
        "usage: GAME [OPTIONS]\n\nPlays game/ beside this executable, or --game DIR.\nDevelopment commands require the full pen CLI.\n\n",
    );
    for command in COMMANDS {
        entry(&mut text, &format!("  {} {}", command.name, command.operands), command.about);
    }
    text.push_str("\nPlayback options:\n");
    for option in OPTIONS.iter().filter(|option| {
        option.commands.is_empty() || (option.place != Place::Command && option.commands.contains(&""))
    }) {
        entry(&mut text, format!("  {} {}", option.name, option.value).trim_end(), option.about);
    }
    text
}

/// How many edits make `a` into `b`.
fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, a) in a.chars().enumerate() {
        let mut previous = row[0];
        row[0] = i + 1;
        for (j, b) in b.iter().enumerate() {
            let replaced = previous + usize::from(a != *b);
            previous = row[j + 1];
            row[j + 1] = replaced.min(previous + 1).min(row[j] + 1);
        }
    }
    row[b.len()]
}

/// ` (did you mean X?)` for the nearest of `names` to `word`, if one is near.
fn suggestion<'a>(word: &str, names: impl Iterator<Item = &'a str>) -> String {
    names
        .map(|name| (distance(word, name), name))
        .filter(|(distance, _)| *distance <= 2)
        .min()
        .map(|(_, name)| format!(" (did you mean {name}?)"))
        .unwrap_or_default()
}

/// The command line, parsed once: a command, its operands and its options,
/// each checked against [`COMMANDS`] and [`OPTIONS`].
#[derive(Debug, Default)]
pub(crate) struct Args {
    /// The command's name; "" plays a packaged game.
    pub(crate) command: &'static str,
    pub(crate) operands: Vec<String>,
    /// The options given, in order, with their values ("" for a flag).
    options: Vec<(&'static str, String)>,
    /// The steps of `--script` and `--script-file`.
    pub(crate) script: Vec<script::Step>,
}

impl Args {
    pub(crate) fn parse(words: impl IntoIterator<Item = String>) -> Result<Args> {
        let mut words = words.into_iter().peekable();
        let mut args = Args::default();
        if let Some(word) = words.next_if(|word| !word.starts_with('-')) {
            let Some(spec) = COMMANDS.iter().find(|spec| spec.name == word) else {
                let names: Vec<String> = COMMANDS.iter().map(|spec| spec.name.to_owned()).collect();
                bail!(
                    "unknown command {word}{}; the commands are {} (pen --help)",
                    suggestion(&word, COMMANDS.iter().map(|spec| spec.name)),
                    listed(&names)
                );
            };
            args.command = spec.name;
        }
        while let Some(word) = words.next() {
            if args.command == "tool" && word == "--" {
                args.operands.extend(words);
                break;
            }
            // A negative number is an operand (`pen remote ADDRESS state -1`).
            let negative = word.strip_prefix('-').is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()));
            if !word.starts_with('-') || negative {
                args.operands.push(word);
                continue;
            }
            let (name, inline) = match word.split_once('=') {
                Some((name, value)) => (name, Some(value.to_owned())),
                None => (word.as_str(), None),
            };
            let name = if name == "-h" { "--help" } else { name };
            let spec = args.option(name)?;
            let value = match (spec.value, inline) {
                ("", None) => String::new(),
                ("", Some(_)) => bail!("{name} takes no value"),
                (_, Some(value)) => value,
                (placeholder, None) => words
                    .next_if(|value| !value.starts_with("--"))
                    .with_context(|| format!("{name} takes a value: {name} {placeholder}"))?,
            };
            args.options.push((spec.name, value));
        }
        let spec = COMMANDS.iter().find(|spec| spec.name == args.command);
        let operands = spec.map_or("", |spec| spec.operands);
        // Each word one operand, those in brackets optional, `...]` any number.
        let words: Vec<&str> = operands.split_whitespace().collect();
        let most = if operands.ends_with("...]") { usize::MAX } else { words.len() };
        let least = words.iter().filter(|word| !word.starts_with('[')).count();
        if let Some(extra) = args.operands.get(most) {
            bail!(
                "{} takes {}, not {extra}",
                invocation(args.command),
                if most == 0 { "no operands".into() } else { format!("{operands} and options") }
            );
        }
        ensure!(args.operands.len() >= least, "{} needs {operands}", invocation(args.command));
        let mut words = args.values("--script");
        for file in args.values("--script-file") {
            let text = std::fs::read_to_string(&file).with_context(|| format!("reading {file}"))?;
            words.extend(text.lines().filter(|line| !line.trim_start().starts_with('#')).map(str::to_owned));
        }
        args.script = script::parse(&words)?;
        Ok(args)
    }

    /// The option `name` of this command, or why there is none.
    fn option(&self, name: &str) -> Result<&'static OptionSpec> {
        let fits = |spec: &OptionSpec| spec.commands.is_empty() || spec.commands.contains(&self.command);
        if let Some(spec) = OPTIONS.iter().find(|spec| spec.name == name && fits(spec)) {
            return Ok(spec);
        }
        let elsewhere: Vec<String> = OPTIONS
            .iter()
            .filter(|spec| spec.name == name)
            .flat_map(|spec| spec.commands.iter().map(|command| invocation(command)))
            .collect();
        if elsewhere.is_empty() {
            let names = OPTIONS.iter().filter(|spec| fits(spec)).map(|spec| spec.name);
            bail!("{} has no option {name}{} (pen --help)", invocation(self.command), suggestion(name, names));
        }
        bail!("{name} is for {}, not {}", listed(&elsewhere), invocation(self.command))
    }

    fn known(name: &str) {
        debug_assert!(OPTIONS.iter().any(|spec| spec.name == name), "no option {name}");
    }

    pub(crate) fn flag(&self, name: &str) -> bool {
        Args::known(name);
        self.options.iter().any(|(option, _)| *option == name)
    }

    /// Every value given to `name`, in order.
    pub(crate) fn values(&self, name: &str) -> Vec<String> {
        Args::known(name);
        self.options.iter().filter(|(option, _)| *option == name).map(|(_, value)| value.clone()).collect()
    }

    /// The last value given to `name`.
    pub(crate) fn value(&self, name: &str) -> Option<String> {
        self.values(name).pop()
    }

    pub(crate) fn path(&self, name: &str) -> Option<PathBuf> {
        self.value(name).map(PathBuf::from)
    }

    pub(crate) fn number<T: std::str::FromStr>(&self, name: &str) -> Result<Option<T>> {
        self.value(name).map(|text| text.parse().map_err(|_| anyhow!("{name} takes a number, not {text}"))).transpose()
    }

    /// The dev tools are asked for: their console, a shot of them, or a
    /// script typing into them.
    pub(crate) fn overlay_asked(&self) -> bool {
        self.flag("--overlay")
            || self.flag("--overlay-shot")
            || self.script.iter().any(|step| matches!(step, script::Step::Overlay(_)))
    }
}

pub(crate) fn size(text: &str) -> Result<(u32, u32)> {
    let (width, height) = text.split_once('x').context("--size takes WIDTHxHEIGHT")?;
    let size = (width.parse()?, height.parse()?);
    if size.0 == 0 || size.1 == 0 {
        bail!("--size must not be empty");
    }
    Ok(size)
}

/// The mods `--mod` names, if any.
pub(crate) fn load_mods(args: &Args) -> Result<Option<mods::Mods>> {
    let directories: Vec<PathBuf> = args.values("--mod").into_iter().map(PathBuf::from).collect();
    if directories.is_empty() {
        return Ok(None);
    }
    mods::Mods::load(&directories).map(Some)
}

#[cfg(all(test, feature = "compiler"))]
mod tests {
    use super::*;

    fn parse(line: &str) -> Result<Args> {
        Args::parse(line.split_whitespace().map(str::to_owned))
    }

    fn error(line: &str) -> String {
        format!("{:#}", parse(line).unwrap_err())
    }

    #[test]
    fn checkout_tools_forward_arguments_after_the_separator_verbatim() {
        let args = parse("tool ink --project ../ink -- generate config.json --write --project ../gdev").unwrap();
        assert_eq!(args.command, "tool");
        assert_eq!(args.value("--project").as_deref(), Some("../ink"));
        assert_eq!(args.operands, ["ink", "generate", "config.json", "--write", "--project", "../gdev"]);
        assert!(parse("tool").is_err());
        assert!(parse("tool ink --unknown").is_err());
    }

    #[test]
    fn arguments_parse_into_a_command_operands_and_options() {
        let args = parse("run --headless --frames 300 --size=640x480 --script wait=2 --script quit").unwrap();
        assert_eq!(args.command, "run");
        assert!(args.flag("--headless") && !args.flag("--mute"));
        assert_eq!(args.number::<usize>("--frames").unwrap(), Some(300));
        assert_eq!(args.value("--size").as_deref(), Some("640x480"));
        assert_eq!(args.values("--script"), ["wait=2", "quit"]);
        assert_eq!(args.script.len(), 2);
        // Without a command, pen plays a package.
        let args = parse("--game dist/gdev -h").unwrap();
        assert_eq!((args.command, args.path("--game")), ("", Some(PathBuf::from("dist/gdev"))));
        assert!(args.flag("--help"));
        assert_eq!(parse("test scenarios.json --only rock --only menu").unwrap().values("--only"), ["rock", "menu"]);
        assert_eq!(parse("test-blot a.blot b.blot --jobs 2").unwrap().operands, ["a.blot", "b.blot"]);
        assert_eq!(parse("packets out --frames 5").unwrap().operands, ["out"]);
        let args = parse("gltf models/a.glb --vat 30 --out assets").unwrap();
        assert_eq!(
            (args.operands.as_slice(), args.number::<f32>("--vat").unwrap()),
            (&["models/a.glb".to_owned()][..], Some(30.0))
        );
        assert!(parse("gltf").is_err() && parse("gltf a.glb --frames 3").is_err());
        assert_eq!(parse("patch v1 v2 --out up.penpatch").unwrap().operands, ["v1", "v2"]);
        assert_eq!(parse("apply up.penpatch").unwrap().operands, ["up.penpatch"]);
        assert!(parse("--version").unwrap().flag("--version"));
        assert!(parse("clean").is_ok());
        let args = parse("push deck.local:7811 --token t --run gdev").unwrap();
        assert_eq!(
            (args.operands.as_slice(), args.value("--run").as_deref()),
            (&["deck.local:7811".to_owned()][..], Some("gdev"))
        );
        assert_eq!(parse("remote 127.0.0.1 watch 100 3 3 5").unwrap().operands.len(), 6);
        assert_eq!(parse("remote 127.0.0.1 state -1").unwrap().operands[2], "-1");
        assert!(parse("serve --listen 0.0.0.0:7811 --allow-run").unwrap().flag("--allow-run"));
        assert!(parse("run --devtools-listen 127.0.0.1:7812").is_ok());
    }

    #[test]
    fn mistakes_are_refused_with_what_was_meant() {
        assert_eq!(
            error("tset"),
            "unknown command tset (did you mean test?); the commands are tool, dev, web, run, build, check, assets, codegen, export, test, bench, packets, test-guest, test-blot, gltf, import-scene, bake-lightmaps, diverge, fuzz, report, new, texture, font, patch, apply, loudness, clean, serve, relay, push, remote and check-workers (pen --help)"
        );
        assert_eq!(error("run --frame 300"), "pen run has no option --frame (did you mean --frames?) (pen --help)");
        assert_eq!(error("run --port 8000"), "--port is for pen web and pen test, not pen run");
        assert_eq!(
            error("build --frames 3"),
            "--frames is for pen, pen dev, pen run, pen bench, pen packets and pen fuzz, not pen build"
        );
        assert_eq!(error("--steam"), "--steam is for pen build, not pen");
        assert_eq!(error("--stem"), "pen has no option --stem (pen --help)");
        assert_eq!(error("dev --overlay=1"), "--overlay takes no value");
        assert_eq!(error("run --frames"), "--frames takes a value: --frames N");
        assert_eq!(error("build --out --web"), "--out takes a value: --out DIR");
        assert_eq!(error("run extra"), "pen run takes no operands, not extra");
        assert_eq!(error("test a.json b.json"), "pen test takes [FILE] and options, not b.json");
        assert_eq!(error("packets"), "pen packets needs DIR");
        assert_eq!(error("patch v1"), "pen patch needs OLD NEW");
        assert_eq!(error("apply a b c"), "pen apply takes PATCH [DIR] and options, not c");
        assert_eq!(error("-x"), "pen has no option -x (pen --help)");
        assert_eq!(error("run --script bogus=1"), "unknown script step bogus=1");
        let args = parse("run --frames many").unwrap();
        assert_eq!(format!("{:#}", args.number::<usize>("--frames").unwrap_err()), "--frames takes a number, not many");
    }

    #[test]
    fn the_dev_tools_are_asked_for_once() {
        assert!(!parse("run --headless --frames 1").unwrap().overlay_asked());
        assert!(parse("run --headless --overlay --frames 1").unwrap().overlay_asked());
        assert!(parse("run --headless --overlay-shot a.png").unwrap().overlay_asked());
        assert!(parse("run --headless --script overlay=/help").unwrap().overlay_asked());
    }

    #[test]
    fn the_usage_tells_of_every_option_and_command() {
        let text = usage();
        for spec in OPTIONS.iter().filter(|spec| spec.place != Place::Prose) {
            assert!(text.contains(&format!("{} {}", spec.name, spec.value).trim_end().to_owned()), "{}", spec.name);
        }
        for spec in COMMANDS {
            assert!(text.contains(&format!("  {}", spec.name)), "{}", spec.name);
        }
        assert!(text.lines().all(|line| line.chars().count() <= 100), "{text}");
        // An option's name means one kind of thing, whichever command it is for.
        for spec in OPTIONS {
            for other in OPTIONS.iter().filter(|other| other.name == spec.name) {
                assert_eq!(spec.value.is_empty(), other.value.is_empty(), "{}", spec.name);
            }
        }
        // Every variable the crate reads is in it.
        for name in [
            "PEN_PRESENT_MODE",
            "PEN_SOFTWARE",
            "PEN_FRAME_TIMES",
            "PEN_RECORD_PCM",
            "PEN_PLUGINS",
            "PEN_ASSET_CACHE",
            "PEN_STARTUP",
            "PEN_WEB_HOST",
            "PEN_STEAM_APP_ID",
            "PEN_CRASH_TEST",
            "BLOT_INLINING",
            "STEAM_BUILD_USER",
            "PEN_STEAMCMD",
            "PEN_BUTLER",
            "BUTLER_API_KEY",
            "PEN_GRAPHICS_TIER",
            "SOURCE_DATE_EPOCH",
            "PEN_REPORTS",
        ] {
            assert!(text.contains(name), "{name}");
        }
    }

    #[test]
    fn names_are_suggested_when_near() {
        assert_eq!(distance("tset", "test"), 2);
        assert_eq!(distance("", "abc"), 3);
        assert_eq!(suggestion("--frame", ["--fps", "--frames"].into_iter()), " (did you mean --frames?)");
        assert_eq!(suggestion("--bogus", ["--fps", "--frames"].into_iter()), "");
    }
}

#[cfg(all(test, not(feature = "compiler")))]
mod player_tests {
    use super::*;

    #[test]
    fn player_accepts_playback_and_updates_but_no_development_commands() {
        for command in ["assets", "dev", "web", "build", "loudness", "fuzz", "patch"] {
            assert!(Args::parse([command.to_owned()]).is_err(), "{command}");
        }
        assert!(Args::parse(["apply".into(), "update.penpatch".into()]).is_ok());
        assert!(Args::parse(["--headless".into(), "--frames".into(), "1".into()]).is_ok());
        assert!(!usage().contains("asset recipes"));
    }
}
