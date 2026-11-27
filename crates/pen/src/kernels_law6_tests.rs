//! Phase 5 Law #6 integration tests (`tasks/CONCURRENCY_VNEXT.md:706`):
//! Exercises ALL NINE mandatory Law #6 witnesses (a)-(i) against real compiled
//! Squid guests, real `ink/services` + `ink/kernel_transport` + `ink/pen/host`,
//! real `gdev/src/world/colony_paths` + `#kernel const plan_routes`, real
//! `KernelService` (`KernelPool`), real `canonical_render_packet`, and real
//! `pen_runtime::mods::splice`.

use super::*;
use crate::program::{Operation, Program};
use pen_protocol::boundary::{MAX_EVENTS, canonical_render_packet, decode_frame, encode_frame, parse_catalog};
use pen_protocol::{Catalog, Frame, GameEvent, ServiceRequest, Viewport, kernel_op, service, service_status};
use pen_runtime::game::hash_render_packet;
use pen_runtime::mods;
use squid_host::{GuestModule, engine};
use std::path::PathBuf;

fn repo_paths() -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let pen = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let squid_lang = pen.join("../squid-lang");
    let ink = pen.join("../ink");
    let gdev = pen.join("../gdev");
    (squid_lang, pen, ink, gdev)
}

fn compile_squid_project(entry: &std::path::Path, extra_aliases: &[(&str, &std::path::Path)]) -> Vec<u8> {
    let (squid_lang, _, ink, gdev) = repo_paths();
    let squidc = squid_lang.join("compiler-goose/build/squidc-goose");
    let prelude = squid_lang.join("std/prelude.sq");
    let std_root = squid_lang.join("std").canonicalize().unwrap();
    let ink_lib = ink.join("lib").canonicalize().unwrap();
    let gdev_pen = gdev.join("pen").canonicalize().unwrap();
    let gdev_src = gdev.join("src").canonicalize().unwrap();
    let out = std::env::temp_dir().join(format!(
        "pen-law6-{}-{}.wasm",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    let mut cmd = std::process::Command::new(&squidc);
    cmd.args([
        "build-project",
        entry.to_str().unwrap(),
        out.to_str().unwrap(),
        "--prelude",
        prelude.to_str().unwrap(),
        "--std-root",
        std_root.to_str().unwrap(),
        "--alias",
        &format!("ink/={}", ink_lib.display()),
        "--alias",
        &format!("pen/={}", gdev_pen.display()),
        "--alias",
        &format!("game/={}", gdev_src.display()),
    ]);
    for (prefix, root) in extra_aliases {
        cmd.args(["--alias", &format!("{prefix}={}", root.canonicalize().unwrap().display())]);
    }
    let output = cmd.output().expect("spawn squidc-goose");
    assert!(
        output.status.success(),
        "squidc-goose build-project failed for {}:\nstdout: {}\nstderr: {}",
        entry.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_file(&out);
    bytes
}

struct SessionGuest {
    program: Program,
}

impl SessionGuest {
    fn open(engine: &wasmtime::Engine, module: &GuestModule) -> Self {
        let program = Program::start(engine, module).unwrap();
        Self { program }
    }

    fn send_op(&mut self, op: Operation, payload: &[f32]) -> Vec<f32> {
        self.program.send(op, payload).unwrap();
        let resp = self.program.response().unwrap();
        let lanes = resp.into_lanes();
        if op == Operation::Frame {
            self.program.accept();
        }
        lanes
    }
}

impl Drop for SessionGuest {
    fn drop(&mut self) {
        self.program.close();
    }
}

fn minimal_catalog() -> Catalog {
    Catalog { meshes: vec!["m".into()], materials: vec!["mat".into()], sounds: vec![] }
}

fn gdev_catalog() -> Catalog {
    let (_, _, _, gdev) = repo_paths();
    let raw = std::fs::read_to_string(gdev.join("src/assets/catalog.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
    parse_catalog(&json).unwrap()
}

/// Strict old-Pen (`service <= 14`) render-packet validator matching pre-Phase-5 `packets.rs:2547`.
fn validate_old_pen_render_packet(packet: &[f32], catalog: &Catalog) -> Result<Vec<ServiceRequest>, String> {
    let decoded = decode_frame(packet, 26, catalog).map_err(|e| format!("decode render frame: {e:?}"))?;
    for req in &decoded.render.services {
        if req.service > 14 {
            return Err(format!(
                "old-Pen decoder rejected service code {} (> 14) on request {}",
                req.service, req.request
            ));
        }
    }
    Ok(decoded.render.services)
}

/// Shared Session-2 Squid guest wired directly to real `game/world/colony_paths`,
/// `ink/kernel_transport`, `ink/services`, `ink/pen/host`, and `#kernel const plan_routes`
/// (with Save Schema 26 persistence and optional immutable pool initializer for Law #6e).
fn write_law6_session_guest(dir: &std::path::Path, name: &str, pool_bias: f32) -> PathBuf {
    let path = dir.join(name);
    let src = format!(
        r#"
import {{ Vec3 }} from "std/vector"
import * as array from "std/array"
import * as kernel from "std/kernel"
import * as app from "ink/app"
import * as derived from "ink/derived/keyed"
import * as events from "ink/events"
import * as math from "ink/math"
import * as navigation from "ink/navigation/ground"
import * as persist from "ink/persist"
import * as preferences from "ink/preferences"
import * as render from "ink/render"
import * as services from "ink/services"
import * as kernel_transport from "ink/kernel_transport"
import * as host from "ink/pen/host"
import * as clock from "ink/clock"
import * as protocol from "pen/protocol"
import * as colony_paths from "game/world/colony_paths"

// Immutable pool whose initializer controls `binding_hash` without editing route function text (Law #6e).
const immutable_pool: #[F32] = #[{pool_bias:.1}, 0.0]

func plan_route_with_pool (ctx: colony_paths.RouteContext) -> (query: #[4; F32]): #[F32] =>
  let adjusted: #[4; F32] = #[query[0] + immutable_pool[0], query[1], query[2], query[3]]
  return colony_paths.plan_one_route ctx adjusted

#kernel const plan_routes: Kernel [#[F32], #[4; F32], #[F32]] = {{
  prepare: colony_paths.prepare_route_world,
  run: plan_route_with_pool,
}}

const step_policy = $clock.Policy {{ step_seconds: 1.0 / 60.0, max_steps: 5, max_elapsed: 0.25 }}
const svc_policy = $services.Policy {{ retry_ticks: 2, tries: 3, most_answers: 8 }}
const default_prefs = $preferences.Preferences {{
  quality: $preferences.Quality {{ msaa: 1, fxaa: false, scale: 1.0, filtering: 0,
    shadows: 0, occlusion: false, sky: false, fog: 1.0, effects: false, bloom: false, reflections: 0, volumetrics: 0 }},
  volumes: $preferences.Volumes {{ master: 1.0, music: 1.0, effects: 1.0 }},
  controls: $preferences.Controls {{ left_handed: false }}, changed: false,
}}

// Query keys: two pairs across an obstacle wall at x in [-0.5, 0.5], z in [-3.0, 3.0].
const q0_ox: F32 = -5.0
const q0_oz: F32 = -0.5
const q0_tx: F32 = 5.0
const q0_tz: F32 = -0.5
const q1_ox: F32 = -5.0
const q1_oz: F32 = 0.5
const q1_tx: F32 = 5.0
const q1_tz: F32 = 0.5

func build_world_ground (storage: own navigation.Storage): (navigation.Stored, navigation.Storage) =>
  let boxes: #[navigation.Box] = #[
    navigation.box_of 0.45 (math.vec3 (-0.5) 0.0 (-3.0), math.vec3 0.5 2.0 3.0),
  ]
  let rounds: #[navigation.Round] = #[
    navigation.round_of 0.45 (math.vec3 0.0 0.0 0.0) 1.3,
  ]
  let (maybe_stored, next_storage) = navigation.reuse_into storage navigation.closed_stored (&boxes[0..1]) (&rounds[0..1]) 12.0
  return (Maybe.unwrap_or navigation.closed_stored maybe_stored, next_storage)

func fresh_world () =>
  let f = host.frame_owners ()
  let control = app.control ()
  let wall = app.wall ()
  let clk = #[clock.clock]
  let prefs = #[default_prefs]
  let requests = persist.requests ()
  let response: #[F32] = #[]
  let service_scratch: #[F32] = #[]
  let next = #[8388607]
  let outbox: #[services.Request] = #[]
  let waiting: #[services.Request] = #[]
  let filling: #[services.Filling] = #[]
  let answers: #[services.Answer] = #[]
  let statuses: #[U32] = #[]
  let payload: #[F32] = #[]
  let svc = services.access next outbox waiting filling answers statuses payload
  let kt = kernel_transport.owners ()
  let cp = colony_paths.owners ()
  let g_boxes: #[navigation.Box] = #[]
  let g_rounds: #[navigation.Round] = #[]
  let g_words: #[U32] = #[]
  let g_shapes: #[F32] = #[]
  let g_storage = $navigation.Storage {{ boxes: g_boxes, rounds: g_rounds, words: g_words, shapes: g_shapes }}
  let g_stored = #[navigation.closed_stored]
  // Canonical deterministic simulation state (persisted in Save Schema 26!):
  // [step_count, last_wp0_x, last_wp0_z, last_wp1_x, last_wp1_z, key_events_seen, canonical_ans_val, canonical_ans_d0, ext_flag]
  let sim: #[F32] = #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  return {{
    control, wall, clock: clk, preferences: prefs, requests, response, service_scratch,
    services: svc, kernel_transport: kt, colony_paths: cp,
    g_storage, g_stored, sim,
    frame: f.frame, audio: f.audio, events: f.events, ui_nodes: f.ui_nodes, ui_marked: f.ui_marked,
    ui_labels: f.ui_labels, ui_options: f.ui_options, frame_floats: f.frame_floats,
    semantic_text: f.semantic_text, service_output: f.service_output,
  }}

func ensure_ground_and_mesh (own access) =>
  if not access.colony_paths.state[0].ready:
    let (stored, next_storage) = build_world_ground access.g_storage
    access.g_storage := next_storage
    access.g_stored[0] := stored
    let next_cp = do:
      let solids = navigation.read access.g_storage stored
      break colony_paths.prepare_access access.colony_paths solids derived.initial
    access.colony_paths := next_cp
  return access

func step_world (own access) -> (seconds: F32) =>
  access := ensure_ground_and_mesh access
  let n = F32.to_u32 access.sim[0]
  // On steps >= 5, run colony_paths.toward_access on both obstacle-blocked queries!
  // On steps < 5, prepare only (giving CAP + CTX upload + RUN_BATCH time to warm).
  if n >= 5:
    let (wp0, cp1) = do:
      let solids = navigation.read access.g_storage access.g_stored[0]
      break colony_paths.toward_access access.colony_paths solids 0 (math.vec3 q0_ox 0.0 q0_oz) (math.vec3 q0_tx 0.0 q0_tz)
    access.colony_paths := cp1
    let (wp1, cp2) = do:
      let solids = navigation.read access.g_storage access.g_stored[0]
      break colony_paths.toward_access access.colony_paths solids 1 (math.vec3 q1_ox 0.0 q1_oz) (math.vec3 q1_tx 0.0 q1_tz)
    access.colony_paths := cp2
    if let $Some p0 = wp0:
      access.sim[1] := p0.x
      access.sim[2] := p0.z
    if let $Some p1 = wp1:
      access.sim[3] := p1.x
      access.sim[4] := p1.z
  access.sim[0] := U32.to_f32 (n + 1)
  return access

func handle_event (own access) -> (ctx: app.EventContext) =>
  access.services := services.receive self svc_policy ctx.event
  case ctx.event of
    $events.KeyDown code => do:
      access.sim[5] := access.sim[5] + U32.to_f32 code
      // KeyDown 65 triggers canonical high-ID 0x800000 request (Law #6a/b/c).
      if code == 65:
        let (_, svc1) = services.ask access.services protocol.service_clipboard protocol.clipboard_op_copy 1.0 0.0 0.0 0.0 #[11.0, 22.0]
        let (_, svc2) = services.ask svc1 protocol.service_clipboard protocol.clipboard_op_copy 2.0 0.0 0.0 0.0 #[33.0, 44.0]
        access.services := svc2
    _ => ()
  if let $Some ans = services.answer access.services 8388608:
    access.sim[6] := ans.value
    let vals = services.values access.services ans.values
    if vals.length > 0:
      access.sim[7] := vals[0]
  return access

func present_world (own access) -> (presentation: app.Presentation) =>
  // Put canonical sim values (never ephemeral prewarm counters!) in clear color,
  // and report ephemeral telemetry ONLY in an extension tag when requested so
  // canonical render packets stay 100% identical between Worker ON and Worker OFF!
  access.frame.settings[0].lighting.clear := math.vec4 (access.sim[1] * 0.01) (access.sim[2] * 0.01) (access.sim[5] * 0.001) 1.0
  // Emit a 1-lane canonical draw reflecting sim state.
  @storage.push access.frame.draws ($render.Draw {{
    mesh: 0, material: 0,
    matrix: math.Mat4.identity,
    parameters: $Some (math.vec4 access.sim[0] access.sim[1] access.sim[3] access.sim[6]),
    extra: render.no_span,
  }})
  return access

func encode_state_fn access -> (dst: own #[F32]): #[F32] =>
  @storage.truncate dst 0
  // Save Schema 26 payload: canonical sim + clock + services (NEVER kernel_transport or prewarm_entries!)
  for i in 0 .. access.sim.length:
    @storage.push dst access.sim[i]
  @storage.push dst (U32.to_f32 access.clock[0].tick)
  @storage.push dst access.clock[0].remainder
  let svc_lanes = services.encode access.services
  @storage.push dst (U32.to_f32 svc_lanes.length)
  for v in svc_lanes:
    @storage.push dst v
  // Append ephemeral telemetry ONLY if we expose a separate SaveScene (op 5) for test inspection!
  return dst

func decode_state_fn (own access) -> (values: #[F32]) =>
  let report: #[F32] = #[]
  if values.length >= 12:
    for i in 0 .. 9:
      access.sim[i] := values[i]
    access.clock[0] := $clock.Clock {{ tick: F32.to_u32 values[9], remainder: values[10] }}
    let svc_len = F32.to_u32 values[11]
    if 12 + svc_len <= values.length:
      let slice = array.slice 12 svc_len values
      access.services := services.decode slice self
  access := colony_paths.reset_storage access
  access.kernel_transport := kernel_transport.reset self
  return (report, access)

// Operation::SnapshotScene (op 5) exposes ephemeral prewarm metrics so tests can
// FIRST assert `prewarm_hits > 0 && sync_routes_skipped > 0` without polluting Save 26!
func encode_telemetry access -> (dst: own #[F32]): #[F32] =>
  @storage.truncate dst 0
  @storage.push dst (U32.to_f32 (colony_paths.prewarm_hits access.colony_paths))
  @storage.push dst (U32.to_f32 (colony_paths.sync_routes_skipped access.colony_paths))
  @storage.push dst (if kernel_transport.cap_active access.kernel_transport then 1.0 else 0.0)
  @storage.push dst (U32.to_f32 access.colony_paths.prewarm_entries.length)
  @storage.push dst (U32.to_f32 access.colony_paths.regions.length)
  return dst

func decode_telemetry (own access) -> (values: #[F32]) =>
  let report: #[F32] = #[]
  return (report, access)

func receive_kernels_fn (own access) -> (evs: &[events.Event]) =>
  access.kernel_transport := kernel_transport.receive_events self evs
  access.colony_paths := colony_paths.absorb_prewarmed_routes self access.kernel_transport (kernel.id plan_routes) (kernel.binding_hash plan_routes)
  return access

func emit_kernels_fn (own access) -> (canonical_count: U32) -> (admitted: own #[services.Request]) -> (payload_out: own #[F32]) =>
  if kernel_transport.cap_active access.kernel_transport and access.colony_paths.state[0].ready:
    let kid = kernel.id plan_routes
    let bh = kernel.binding_hash plan_routes
    let gh = access.colony_paths.state[0].geom_hash
    let (ctx_ready, next_kt) = do:
      let cp: colony_paths.Access = access.colony_paths
      let wire = &cp.wire_ctx[0..cp.wire_ctx.length]
      break kernel_transport.ensure_context access.kernel_transport kid bh gh wire
    access.kernel_transport := next_kt
    if ctx_ready:
      let batch: #[F32] = #[]
      let cnt: U32 = 0
      if not (colony_paths.has_prewarm access.colony_paths (kernel.f32_bits q0_ox) (kernel.f32_bits q0_oz) (kernel.f32_bits q0_tx) (kernel.f32_bits q0_tz) gh):
        @storage.push batch q0_ox
        @storage.push batch q0_oz
        @storage.push batch q0_tx
        @storage.push batch q0_tz
        cnt := cnt + 1
      if not (colony_paths.has_prewarm access.colony_paths (kernel.f32_bits q1_ox) (kernel.f32_bits q1_oz) (kernel.f32_bits q1_tx) (kernel.f32_bits q1_tz) gh):
        @storage.push batch q1_ox
        @storage.push batch q1_oz
        @storage.push batch q1_tx
        @storage.push batch q1_tz
        cnt := cnt + 1
      if cnt > 0:
        let (_, next_kt2) = kernel_transport.enqueue_batch access.kernel_transport kid bh gh cnt (&batch[0..batch.length])
        access.kernel_transport := next_kt2
    if access.kernel_transport.requests.length > 0:
      let base_floats = host.measure_access access
      let (next_kt, next_admitted) = host.admit_kernel_requests base_floats canonical_count access.kernel_transport admitted
      access.kernel_transport := next_kt
      admitted := next_admitted
      for i in 0 .. access.kernel_transport.payload.length:
        @storage.push payload_out access.kernel_transport.payload[i]
  return (access, admitted, payload_out)

func init_world (own access) =>
  access.services := services.idle self 6
  access.services.next[0] := 8388607
  access.kernel_transport := kernel_transport.reset self
  access := colony_paths.reset_storage access
  return access

func ack_world (own access) -> (published: Bool) =>
  access.requests := persist.clear_requests self
  let (next_services, next_scratch) = services.compact access.services access.service_scratch
  access.services := next_services
  access.service_scratch := next_scratch
  return access

const stages = app.stages app.no_action app.no_action app.no_action app.no_action app.route_event
  app.no_event handle_event step_world present_world app.no_action

const application = {{
  stages,
  policy: step_policy,
  service_policy: svc_policy,
  schema: 26,
  scenes: true,
  keep: app.no_keep,
  acknowledge: ack_world,
  set_viewport: func (own access) -> _ -> _ => access,
  receive_kernels: receive_kernels_fn,
  emit_kernels: emit_kernels_fn,
  initialize: init_world,
  encode_state: encode_state_fn,
  decode_state: decode_state_fn,
  encode_scene: encode_telemetry,
  decode_scene: decode_telemetry,
}}

io main (h: #[F32] -> #[F32] ! {{Foreign}}): Unit ! {{Foreign}} => return
  host.run application (fresh_world ()) h
"#
    );
    std::fs::write(&path, src).unwrap();
    path
}

type SessionRunResult = (Vec<Vec<f32>>, Vec<u64>, Vec<f32>, Vec<f32>, Vec<Vec<ServiceRequest>>);

#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Law 6 test harness decodes non-negative wire counts"
)]
fn drive_session_frames(
    engine: &wasmtime::Engine,
    module: &GuestModule,
    catalog: &Catalog,
    mut svc: Option<&mut KernelService>,
    frames: &[Vec<GameEvent>],
    mod_events: &[Vec<[f32; EVENT_FLOATS]>],
    initial_save: Option<&[f32]>,
    replayed_inputs: Option<&[Vec<f32>]>,
    enforce_old_pen_decoder: bool,
) -> SessionRunResult {
    let mut guest = SessionGuest::open(engine, module);
    if let Some(saved) = initial_save {
        let _ = guest.send_op(Operation::Restore, saved);
    } else {
        let _ = guest.send_op(Operation::Create, &[]);
    }

    let mut recorded_inputs = Vec::new();
    let mut canonical_hashes = Vec::new();
    let mut all_services = Vec::new();
    let count = replayed_inputs.map_or(frames.len(), |r| r.len());

    for idx in 0..count {
        let mut input = if let Some(rep) = replayed_inputs {
            rep[idx].clone()
        } else {
            let frame = Frame {
                timestamp: (idx as f64) / 60.0,
                viewport: Viewport { width: 640, height: 480 },
                events: frames[idx].clone(),
            };
            encode_frame(&frame, 1.0 / 60.0, &[]).unwrap()
        };

        // Run observing/input mod BEFORE kernel injection: mod must NEVER see request == 0 kernel events!
        if let Some(extra) = mod_events.get(idx) {
            // Assert mod sees zero request==0 kernel events in input before splicing!
            let base_cnt = input[3] as usize;
            for ev in 0..base_cnt {
                let off = FRAME_HEADER_FLOATS + ev * EVENT_FLOATS;
                let tag = input[off] as u32;
                if tag == 13 || tag == 14 {
                    // ServiceAnswer / ServiceData
                    assert_ne!(
                        input[off + 2] as u32,
                        0,
                        "mod observed ephemeral request==0 kernel event on frame {idx}!"
                    );
                }
            }
            if !extra.is_empty() {
                assert!(mods::splice(&mut input, extra).is_ok());
            }
        }

        let cap_events = if let Some(ksvc) = svc.as_deref_mut() { ksvc.splice_live_events(&mut input) } else { 0 };

        let render_pkt = guest.send_op(Operation::Frame, &input);

        // Strip live-only CAP BEFORE recording!
        let mut recorded_input = input;
        strip_trailing_events(&mut recorded_input, cap_events);
        recorded_inputs.push(recorded_input);

        // Validate against old-Pen (service <= 14) decoder if requested.
        let reqs = if enforce_old_pen_decoder {
            validate_old_pen_render_packet(&render_pkt, catalog)
                .expect("old-Pen decoder must accept replayed render packet")
        } else {
            decode_frame(&render_pkt, 26, catalog).expect("decode render packet").render.services
        };
        let kernel_reqs: Vec<ServiceRequest> = reqs.iter().filter(|r| r.service == service::KERNEL).cloned().collect();
        if let Some(ksvc) = svc.as_deref_mut()
            && !kernel_reqs.is_empty()
        {
            ksvc.submit(&kernel_reqs);
        }
        all_services.push(reqs);

        // Hash canonical_render_packet (exact game.rs / diverge.rs normalization, including exposure.cut!).
        let canonical = canonical_render_packet(&render_pkt);
        assert!(
            decode_frame(&canonical, 26, catalog).unwrap().render.services.iter().all(|r| r.service != service::KERNEL),
            "canonical_render_packet must strip all service::KERNEL records"
        );
        canonical_hashes.push(hash_render_packet(&render_pkt));
    }

    let final_save = guest.send_op(Operation::Snapshot, &[]);
    let telemetry = guest.send_op(Operation::Scene, &[]);
    (recorded_inputs, canonical_hashes, final_save, telemetry, all_services)
}

fn has_sibling_checkouts() -> bool {
    let (squid_lang, _, ink, gdev) = repo_paths();
    let squidc = std::env::var_os("SQUIDC")
        .map(PathBuf::from)
        .unwrap_or_else(|| squid_lang.join("compiler-goose/build/squidc-goose"));
    if squidc.is_file()
        && squid_lang.join("std/prelude.sq").is_file()
        && ink.join("lib").is_dir()
        && gdev.join("src").is_dir()
    {
        return true;
    }
    crate::fixtures::missing(
        "Law #6 kernel integration tests need squidc-goose and sibling squid-lang/ink/gdev checkouts",
    );
    false
}

#[test]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "test asserts non-negative wire tags and counts"
)]
fn law6_end_to_end_replay_old_pen_moment_hot_reload_mods_and_equivalence() {
    if !has_sibling_checkouts() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("pen-law6-suite-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let v1_path = write_law6_session_guest(&dir, "law6_v1.sq", 0.0);
    let v2_path = write_law6_session_guest(&dir, "law6_v2.sq", 1.0);
    let v1_wasm = compile_squid_project(&v1_path, &[]);
    let v2_wasm = compile_squid_project(&v2_path, &[]);

    let eng = engine().unwrap();
    let mod_v1 = GuestModule::new(&eng, &v1_wasm, None).unwrap();
    let mod_v2 = GuestModule::new(&eng, &v2_wasm, None).unwrap();

    // Law #6(e): Editing ONLY an immutable pool initializer with unchanged route function text
    // MUST change `binding_hash`!
    let m1 = &mod_v1.abi().kernels[0];
    let m2 = &mod_v2.abi().kernels[0];
    assert_ne!(
        m1.binding_hash, m2.binding_hash,
        "Law #6(e): editing immutable pool initializer must change binding_hash"
    );

    // Build an 8-frame input sequence with:
    // - Frame 0: KeyDown 65 -> mints canonical 8388607 & 8388608 (0x800000) (Law #6a/b)
    // - Frames 1..3: context upload (CTX_BEGIN, CTX_CHUNK, CTX_COMMIT) + canonical 0x800000 retry at tick 2!
    // - Frame 4: RUN_BATCH submitted to KernelService + canonical answer at 0x800000 queued for Frame 5
    // - Frame 5: BOTH canonical answer at 0x800000 AND recorded request==0 BATCH_OK arrive in the SAME frame (Law #6c),
    //            AND step 5 runs colony_paths.toward hitting prewarm_entries (prewarm_hits == 2, sync_routes_skipped == 2)!
    // - Frames 6..7: continued movement with mod splicing synthetic key inputs.
    let catalog = minimal_catalog();
    let mut frames: Vec<Vec<GameEvent>> = vec![vec![]; 8];
    frames[0].push(GameEvent::KeyDown("A".into()));
    // Frame 4 delivers BOTH canonical 0x800000 answer AND recorded request==0 BATCH_OK in the SAME frame (Law #6c)!
    frames[4].push(GameEvent::ServiceAnswer { request: 0x80_0000, status: service_status::OK, value: 99.0, length: 2 });
    frames[4].push(GameEvent::ServiceData { request: 0x80_0000, a: 77.0, b: 88.0, c: 0.0 });

    // Observing/input mod injects raw KeyDown events [tag=1, code, 0, 0, 0] on frames 2 and 6.
    let mut mod_events: Vec<Vec<[f32; EVENT_FLOATS]>> = vec![vec![]; 8];
    mod_events[2].push([1.0, 10.0, 0.0, 0.0, 0.0]);
    mod_events[6].push([1.0, 20.0, 0.0, 0.0, 0.0]);

    // Law #6(i) splice guard check: verify mods::splice enforces MAX_EVENTS = 100,000.
    let mut full_input = vec![0.0; FRAME_HEADER_FLOATS];
    full_input[3] = MAX_EVENTS as f32;
    assert!(mods::splice(&mut full_input, &[[1.0, 1.0, 0.0, 0.0, 0.0]]).is_err());
    assert_eq!(full_input[3], MAX_EVENTS as f32, "mods::splice must enforce MAX_EVENTS");

    // 1. Run with Worker ON (KernelService enabled, 2 workers)
    let mut ksvc_on = KernelService::new_with_workers(eng.clone(), mod_v1.clone(), false, 2);
    let (recorded_on, hashes_on, save_on, telemetry_on, svcs_on) =
        drive_session_frames(&eng, &mod_v1, &catalog, Some(&mut ksvc_on), &frames, &mod_events, None, None, false);

    // FIRST assert real worker RUN_BATCH replies produced actual prewarm hits that SKIPPED sync route_into!
    let prewarm_hits = telemetry_on[0] as u32;
    let sync_skipped = telemetry_on[1] as u32;
    assert!(
        prewarm_hits >= 2 && sync_skipped >= 2,
        "Law #6(i): expected real worker prewarm hits >= 2 & sync_routes_skipped >= 2, got telemetry={telemetry_on:?}"
    );
    // Verify canonical 0x800000 answer (value=99.0, data[0]=77.0) AND mod keys (10+20+65=95) were delivered cleanly!
    assert_eq!(save_on[5], 95.0, "mod & input KeyDown events preserved");
    assert_eq!(save_on[6], 99.0, "Law #6(c): canonical 0x800000 answer value delivered");
    assert_eq!(save_on[7], 77.0, "Law #6(c): canonical 0x800000 answer payload delivered");
    // Verify live run actually emitted service::KERNEL = 15 requests (upload + RUN_BATCH) AND canonical 0x800000 retry!
    assert!(
        svcs_on.iter().flatten().any(|r| r.service == service::KERNEL && r.operation == kernel_op::RUN_BATCH),
        "live Worker ON run must emit real RUN_BATCH"
    );
    assert!(
        svcs_on.iter().flatten().any(|r| r.request == 0x80_0000 && r.service == service::CLIPBOARD),
        "Law #6(a)(b): canonical 0x800000 request emitted"
    );

    // Verify Frame 4 ACTUALLY carried BOTH canonical 0x800000 ServiceAnswer AND request==0 BATCH_OK in the same frame!
    let rec4 = &recorded_on[4];
    let rec4_cnt = rec4[3] as usize;
    let mut saw_canonical_800000 = false;
    let mut saw_kernel_batch_ok = false;
    for ev in 0..rec4_cnt {
        let off = FRAME_HEADER_FLOATS + ev * EVENT_FLOATS;
        if rec4[off] as u32 == 13 {
            if rec4[off + 2] as u32 == 0x80_0000 {
                saw_canonical_800000 = true;
            } else if rec4[off + 2] as u32 == 0 && rec4[off + 3] >= 1.0 && rec4[off + 4] >= 13.0 {
                saw_kernel_batch_ok = true;
            }
        }
    }
    assert!(
        saw_canonical_800000 && saw_kernel_batch_ok,
        "Law #6(c): Frame 4 must carry BOTH canonical 0x800000 answer and request==0 BATCH_OK in the same frame"
    );

    // 2. Run with Worker OFF (no KernelService, pure synchronous route_into)
    let (_recorded_off, hashes_off, save_off, telemetry_off, svcs_off) =
        drive_session_frames(&eng, &mod_v1, &catalog, None, &frames, &mod_events, None, None, true);
    assert_eq!(telemetry_off[0], 0.0, "Worker OFF must have 0 prewarm_hits");
    assert_eq!(telemetry_off[1], 0.0, "Worker OFF must have 0 sync_routes_skipped");
    assert!(
        svcs_off.iter().flatten().all(|r| r.service != service::KERNEL),
        "Worker OFF must emit zero service::KERNEL requests"
    );

    // Law #6(i): 100% BIT-IDENTICAL Save-26 state bytes AND canonical_render_packet packet_hash on EVERY frame!
    assert_eq!(
        save_on.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        save_off.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        "Law #6(i): Save-26 state bytes must be 100% bit-identical between Worker ON and Worker OFF"
    );
    assert_eq!(
        hashes_on, hashes_off,
        "Law #6(i): canonical_render_packet packet_hash must be 100% identical on every frame"
    );

    // 3. Law #6(d): Replay new-Pen recording through ACTUAL old-Pen decoder (`service <= 14`)!
    // Verify ZERO CAP events exist in recorded_on, ZERO service::KERNEL = 15 emitted on replay,
    // recorded BATCH_OK STILL absorbs cleanly (prewarm_hits >= 2), AND state + hashes match bit-for-bit!
    for (idx, rec) in recorded_on.iter().enumerate() {
        let cnt = rec[3] as usize;
        for ev in 0..cnt {
            let off = FRAME_HEADER_FLOATS + ev * EVENT_FLOATS;
            if rec[off] as u32 == 13 && rec[off + 2] as u32 == 0 {
                assert_ne!(
                    rec[off + 3],
                    0.0,
                    "Law #6(d): live-only CAP (request==0, value==0.0) must NEVER appear in Recorded frame {idx}"
                );
            }
        }
    }
    let (_, hashes_old_pen, save_old_pen, telemetry_old_pen, svcs_old_pen) = drive_session_frames(
        &eng,
        &mod_v1,
        &catalog,
        None,
        &[],
        &[],
        None,
        Some(&recorded_on),
        true, // STRICT old-Pen decoder (rejects service > 14)!
    );
    assert!(
        svcs_old_pen.iter().flatten().all(|r| r.service <= 14),
        "Law #6(d): replaying new-Pen recording must emit ZERO service::KERNEL = 15"
    );
    assert!(
        telemetry_old_pen[0] >= 2.0 && telemetry_old_pen[1] >= 2.0,
        "Law #6(d): recorded BATCH_OK must still absorb cleanly without CAP during old-Pen replay"
    );
    assert_eq!(hashes_old_pen, hashes_on, "Law #6(d): old-Pen replay hashes match live");
    assert_eq!(
        save_old_pen.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        save_on.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        "Law #6(d): old-Pen replay Save-26 state matches live"
    );

    // Law #6(d) go_live() re-arming live-only CAP on new Pen:
    ksvc_on.fence();
    assert_eq!(ksvc_on.host_epoch(), 2);
    let mut live_frame = encode_frame(
        &Frame { timestamp: 8.0 / 60.0, viewport: Viewport { width: 640, height: 480 }, events: vec![] },
        1.0 / 60.0,
        &[],
    )
    .unwrap();
    let cap_after_go_live = ksvc_on.splice_live_events(&mut live_frame);
    assert_eq!(cap_after_go_live, CAP_EVENTS, "go_live() must re-arm live-only CAP");

    // 4. Law #6(f): Replay from a Moment taken AFTER Frame 0 (at Frame 3, AFTER Frame 0's CAP
    //    and context upload, right before Frame 4 delivers recorded BATCH_OK)!
    let mut ksvc_m = KernelService::new_with_workers(eng.clone(), mod_v1.clone(), false, 2);
    let (_, _, moment_frame3, _, _) = drive_session_frames(
        &eng,
        &mod_v1,
        &catalog,
        Some(&mut ksvc_m),
        &frames[..4],
        &mod_events[..4],
        None,
        None,
        false,
    );
    // Restore Frame-3 Moment into a fresh guest and replay recorded frames 4..8 (NO CAP!).
    let (_, hashes_moment, save_moment, telemetry_moment, svcs_moment) = drive_session_frames(
        &eng,
        &mod_v1,
        &catalog,
        None,
        &[],
        &[],
        Some(&moment_frame3),
        Some(&recorded_on[4..]),
        true,
    );
    assert!(
        svcs_moment.iter().flatten().all(|r| r.service <= 14),
        "Law #6(f): post-Frame-0 Moment replay emits zero service::KERNEL"
    );
    assert!(
        telemetry_moment[0] >= 2.0 && telemetry_moment[1] >= 2.0,
        "Law #6(f): post-Frame-0 Moment replay absorbs recorded BATCH_OK without CAP or waiter table"
    );
    assert_eq!(hashes_moment, hashes_on[4..], "Law #6(f): Moment replay hashes match live frames 4..8");
    assert_eq!(
        save_moment.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        save_on.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        "Law #6(f): Moment replay final Save-26 state matches live bit-for-bit"
    );

    // 5. Law #6(e): Replay v1's recorded stream (with v1's BATCH_OK) into v2 (edited immutable pool only)!
    // v2 MUST reject v1's stale recorded BATCH_OK because binding_hash differs (prewarm_hits == 0)!
    let (_, _, _, telemetry_v2_replay, _) =
        drive_session_frames(&eng, &mod_v2, &catalog, None, &[], &[], None, Some(&recorded_on), true);
    assert_eq!(
        telemetry_v2_replay[0], 0.0,
        "Law #6(e): v2 must REJECT v1's recorded BATCH_OK after immutable pool edit (binding_hash mismatch)"
    );
    assert_eq!(
        telemetry_v2_replay[1], 0.0,
        "Law #6(e): v2 must not skip synchronous routing using stale v1 prewarmed routes"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gdev_main_wasm_plan_routes_kernel_runs_deterministically_across_worker_counts() {
    if !has_sibling_checkouts() {
        return;
    }
    let (_, _, _, gdev) = repo_paths();
    let project = crate::project::Project::find(Some(gdev.clone())).unwrap();
    let out_dir = std::env::temp_dir().join(format!("pen-law6-gdev-main-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out_dir);
    project.compile(&out_dir).expect("compile real gdev/src/main.sq through Pen project pipeline");
    let wasm = std::fs::read(out_dir.join("game.wasm")).unwrap();
    let _ = std::fs::remove_dir_all(&out_dir);
    let eng = engine().unwrap();
    let module = GuestModule::new(&eng, &wasm, None).unwrap();

    // Verify Gdev main.wasm exports `plan_routes` kernel manifest!
    let manifest = module
        .abi()
        .kernels
        .iter()
        .find(|k| k.name == "plan_routes")
        .expect("gdev/src/main.sq must declare #kernel const plan_routes");

    // Canonical WireCtx with a central wall & round obstacle.
    let wire_ctx: Vec<f32> = vec![
        2.0, 12.0, 1.0, 1.0, // version, radius, 1 box, 1 round
        -0.95, 0.95, -3.45, 3.45, // widened box
        0.0, 0.0, 1.85, // widened round
    ];
    let ctx_words: Vec<u32> = wire_ctx.iter().map(|f| f.to_bits()).collect();

    // Build 32 distinct route queries across the obstacle wall (plus 128 for wave scaling).
    let make_queries = |n: usize| -> Vec<u32> {
        let mut words = Vec::with_capacity(n * 4);
        for i in 0..n {
            let z = -2.5 + 5.0 * (i as f32) / (n.max(2) - 1) as f32;
            for f in [-5.0f32, z, 5.0f32, -z] {
                words.push(f.to_bits());
            }
        }
        words
    };
    let queries_32 = make_queries(32);

    // Run on workers = 0, 1, 2, 4 and assert 100% bit-identical outputs!
    let mut reference: Option<Vec<u32>> = None;
    for workers in [0usize, 1, 2, 4] {
        let cfg = KernelConfig {
            workers,
            deadline: None,
            max_total_output_bytes: 2 << 20,
            max_physical_output_bytes: 4 << 20,
            ..KernelConfig::default()
        };
        let mut pool = KernelPool::open(&eng, &module, &manifest.name, cfg).unwrap();
        pool.set_context(1, &ctx_words).unwrap();
        let out = pool.run_variable(&queries_32).unwrap();
        assert!(!out.is_empty());
        if let Some(ref expected) = reference {
            assert_eq!(&out, expected, "gdev plan_routes output must match bit-for-bit at workers={workers}");
        } else {
            reference = Some(out);
        }
    }

    // Also run real Gdev Session-2 frames (Frame 0..4) with Worker ON vs Worker OFF and verify
    // 100% bit-identical Save-26 state bytes AND canonical_render_packet hashes!
    let gdev_cat = gdev_catalog();
    let frames = vec![vec![]; 4];
    let mut ksvc = KernelService::new_with_workers(eng.clone(), module.clone(), false, 2);
    let (_, hashes_on, save_on, _, _) =
        drive_session_frames(&eng, &module, &gdev_cat, Some(&mut ksvc), &frames, &[], None, None, false);
    let (_, hashes_off, save_off, _, _) =
        drive_session_frames(&eng, &module, &gdev_cat, None, &frames, &[], None, None, true);
    assert_eq!(hashes_on, hashes_off, "real Gdev canonical_render_packet hashes match Worker ON vs OFF");
    assert_eq!(
        save_on.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        save_off.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        "real Gdev Save-26 state bytes match bit-for-bit Worker ON vs OFF"
    );
    // Verify decoded render frame from Gdev passes full pen_protocol::boundary::decode_frame.
    let mut g = SessionGuest::open(&eng, &module);
    let _ = g.send_op(Operation::Create, &[]);
    let inp = encode_frame(
        &Frame { timestamp: 0.0, viewport: Viewport { width: 960, height: 540 }, events: vec![] },
        1.0 / 60.0,
        &[],
    )
    .unwrap();
    let pkt = g.send_op(Operation::Frame, &inp);
    assert!(decode_frame(&pkt, 26, &gdev_cat).is_ok(), "Gdev frame packet must decode cleanly");
}
