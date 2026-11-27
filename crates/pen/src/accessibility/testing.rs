//! Role queries and user actions for a live headless session.
#![allow(dead_code, reason = "testing entry points for the native binary's session")]

use anyhow::{Error, Result};
use pen_protocol::GameEvent;
use pen_runtime::role_testing::{ActionError, IntoRole, RoleNode, RoleScreen};

use crate::session::Session;

/// Drives a session one step at a time and queries its latest presented interface.
pub struct SessionTester<'a> {
    session: &'a mut Session,
}

impl<'a> SessionTester<'a> {
    pub fn new(session: &'a mut Session) -> Self {
        Self { session }
    }

    pub fn screen(&self) -> RoleScreen<'_> {
        RoleScreen::new(self.session.interface())
    }

    /// Takes one session step and returns whether the session should continue.
    pub fn step(&mut self) -> Result<bool> {
        self.session.step(None)
    }

    /// Queues input and takes one session step.
    pub fn dispatch(&mut self, event: GameEvent) -> Result<bool> {
        self.session.input.enqueue(event);
        self.step()
    }

    /// Activates a matching node. Role-driven actions return the target as it
    /// appeared before the action, even if the next frame removes it.
    pub fn click_by_role(&mut self, role: impl IntoRole, name: &str) -> Result<RoleNode> {
        self.act(role, name, |screen, node| screen.click(node))
    }

    pub fn focus_by_role(&mut self, role: impl IntoRole, name: &str) -> Result<RoleNode> {
        self.act(role, name, |screen, node| screen.focus(node))
    }

    pub fn increment_by_role(&mut self, role: impl IntoRole, name: &str) -> Result<RoleNode> {
        self.act(role, name, |screen, node| screen.increment(node))
    }

    pub fn decrement_by_role(&mut self, role: impl IntoRole, name: &str) -> Result<RoleNode> {
        self.act(role, name, |screen, node| screen.decrement(node))
    }

    pub fn set_value_by_role(&mut self, role: impl IntoRole, name: &str, value: f32) -> Result<RoleNode> {
        self.act(role, name, |screen, node| screen.set_value(node, value))
    }

    pub fn expand_by_role(&mut self, role: impl IntoRole, name: &str) -> Result<RoleNode> {
        self.act(role, name, |screen, node| screen.expand(node))
    }

    pub fn collapse_by_role(&mut self, role: impl IntoRole, name: &str) -> Result<RoleNode> {
        self.act(role, name, |screen, node| screen.collapse(node))
    }

    pub fn assert_audit_clean(&self) -> Result<()> {
        self.screen().assert_audit_clean().map_err(Error::msg)
    }

    fn act(
        &mut self,
        role: impl IntoRole,
        name: &str,
        action: impl FnOnce(&RoleScreen<'_>, &RoleNode) -> Result<GameEvent, ActionError>,
    ) -> Result<RoleNode> {
        let screen = self.screen();
        let target = screen.get_by_role_and_name(role, name)?;
        let event = action(&screen, &target)?;
        self.dispatch(event)?;
        Ok(target)
    }
}

#[cfg(all(test, feature = "compiler", feature = "devtools"))]
mod tests {
    use std::path::Path;

    use pen_protocol::boundary::{
        EVENT_FLOATS, FRAME_HEADER_FLOATS, PACKET_HEADER_FLOATS, PACKET_MAGIC, UI_NODE_FLOATS, event_tags,
        lanes::{header, ui_node},
    };
    use pen_protocol::{Catalog, ui_action, ui_role, ui_state};
    use pen_runtime::role_testing::QueryError;
    use squid_host::GuestModule;

    use super::*;
    use crate::session::{Gpu, Options, Paths};
    use crate::squid_test::tests::{program, signed_leb};

    const DIALOG: u32 = 10;
    const SLIDER: u32 = 11;
    const OPTION: u32 = 13;
    const OUTSIDE: u32 = 18;
    const DISABLED: u32 = 20;

    fn push_node(packet: &mut Vec<f32>, id: u32, parent: u32, role: u32, name: &str) -> usize {
        let start = packet.len();
        let mut row = [0.0; UI_NODE_FLOATS];
        row[ui_node::ID] = id as f32;
        row[ui_node::PARENT] = parent as f32;
        row[ui_node::ROLE] = role as f32;
        row[ui_node::WIDTH] = 32.0;
        row[ui_node::HEIGHT] = 24.0;
        row[ui_node::NAME_LENGTH] = name.chars().count() as f32;
        row[ui_node::TEXT_LENGTH] = row[ui_node::NAME_LENGTH];
        packet.extend(row);
        packet.extend(name.chars().map(|c| u32::from(c) as f32));
        packet[1 + header::UI_COUNT] += 1.0;
        packet[1 + header::UI_LANES] += (packet.len() - start) as f32;
        start
    }

    fn fixture() -> Vec<u8> {
        const HANDSHAKE: i32 = 2048;
        const CREATE: i32 = 2080;
        const SNAPSHOT: i32 = 2160;
        const RESTORE: i32 = 2240;
        const RENDER: i32 = 8192;

        let mut render = vec![0.0; 1 + PACKET_HEADER_FLOATS];
        render[0] = 2.0; // Frame reply
        render[1] = PACKET_MAGIC;
        render[2] = 1.0; // Packet schema
        for start in [header::VIEW, header::PROJECTION, header::SHADOW_VIEW] {
            for diagonal in [0, 5, 10, 15] {
                render[1 + start + diagonal] = 1.0;
            }
        }
        for at in [
            header::CLEAR + 3,
            header::LIGHT_DIRECTION + 1,
            header::QUALITY_MSAA,
            header::QUALITY_SCALE,
            header::FOG_HEIGHT,
            header::FOG_NOISE_SCALE,
            header::EXPOSURE_HIGH,
            header::EXPOSURE_RANGE_HIGH,
            header::EXPOSURE_TARGET,
            header::GRADE_PIVOT,
            header::GRADE_GAMMA,
            header::GRADE_GAMMA + 1,
            header::GRADE_GAMMA + 2,
        ] {
            render[1 + at] = 1.0;
        }

        let dialog = push_node(&mut render, DIALOG, 0, ui_role::DIALOG, "Settings");
        render[dialog + ui_node::MODAL] = 1.0;
        let slider = push_node(&mut render, SLIDER, 1, ui_role::SLIDER, "Music Volume");
        render[slider + ui_node::FOCUSABLE] = 1.0;
        render[slider + ui_node::FOCUSED] = 1.0;
        render[slider + ui_node::ACTIONS] = ((1 << ui_action::INCREMENT)
            | (1 << ui_action::DECREMENT)
            | (1 << ui_action::SET_VALUE)
            | (1 << ui_action::EXPAND)
            | (1 << ui_action::COLLAPSE)) as f32;
        render[slider + ui_node::EXPANDED] = ui_state::FALSE as f32;
        render[slider + ui_node::VALUE] = 0.5;
        render[slider + ui_node::MAXIMUM] = 1.0;
        render[slider + ui_node::STEP] = 0.25;
        render[slider + ui_node::DESCRIPTION_LENGTH] = 5.0;
        render[slider + ui_node::TEXT_LENGTH] += 9.0;
        render[1 + header::UI_LANES] += 9.0;
        render.extend("AudioHalf".chars().map(|c| u32::from(c) as f32));
        push_node(&mut render, 12, 1, ui_role::LIST, "Quality");
        let option = push_node(&mut render, OPTION, 3, ui_role::LISTITEM, "Low");
        render[option + ui_node::FOCUSABLE] = 1.0;
        render[option + ui_node::SELECTED] = ui_state::TRUE as f32;
        render[option + ui_node::ACTIONS] = (1 << ui_action::ACTIVATE) as f32;
        for (id, name, state) in
            [(14, "All", ui_state::TRUE), (15, "None", ui_state::FALSE), (16, "Some", ui_state::MIXED)]
        {
            let checkbox = push_node(&mut render, id, 1, ui_role::CHECKBOX, name);
            render[checkbox + ui_node::CHECKED] = state as f32;
        }
        let outside = push_node(&mut render, OUTSIDE, 0, ui_role::BUTTON, "Outside");
        render[outside + ui_node::ACTIONS] = (1 << ui_action::ACTIVATE) as f32;
        let disabled = push_node(&mut render, 19, 1, ui_role::GROUP, "Unavailable");
        render[disabled + ui_node::DISABLED] = 1.0;
        let button = push_node(&mut render, DISABLED, 9, ui_role::BUTTON, "Apply");
        render[button + ui_node::ACTIONS] = (1 << ui_action::ACTIVATE) as f32;
        let receipt = push_node(&mut render, 21, 1, ui_role::PROGRESSBAR, "Received action");
        render[receipt + ui_node::MINIMUM] = -1.0;
        render[receipt + ui_node::MAXIMUM] = 10000.0;
        let unnamed = push_node(&mut render, 22, 1, ui_role::BUTTON, "");
        render[unnamed + ui_node::HIDDEN] = 1.0;

        let integer = |value| [vec![0x41], signed_leb(value)].concat(); // i32.const
        let float = |value: f32| [vec![0x43], value.to_le_bytes().to_vec()].concat(); // f32.const
        let load = |at| [integer(at), vec![0x28, 2, 0]].concat(); // i32.load
        let store = |at, value| [integer(at), value, vec![0x36, 2, 0]].concat(); // i32.store
        let request = |offset| [load(0), vec![0x2a, 2, offset, 0xa8]].concat(); // f32.load; i32.trunc_f32_s
        let exchange = || [integer(0), vec![0x20, 0], load(20), vec![0x10, 0, 0x36, 2, 0]].concat();
        let lane = |index| RENDER + 4 + i32::try_from(index * 4).unwrap(); // array length precedes the reply
        let mut main = [store(20, integer(HANDSHAKE)), exchange(), vec![0x03, 0x40]].concat(); // loop
        main.extend([request(8), vec![0x45, 0x04, 0x40], integer(0), vec![0x0f, 0x0b]].concat()); // Stop
        for (operation, reply) in [(1, CREATE), (3, SNAPSHOT), (4, RESTORE)] {
            main.extend(
                [
                    request(8),
                    integer(operation),
                    vec![0x46, 0x04, 0x40], // i32.eq; if
                    store(20, integer(reply)),
                    exchange(),
                    vec![0x0c, 1, 0x0b], // restart the loop
                ]
                .concat(),
            );
        }

        // Frame: walk input events. Echo a received UI action as
        // node * 100 + action * 10 + value, or -1 when there was no action.
        // A blur also reveals the unnamed button for the audit assertion.
        let first_event = i32::try_from(12 + FRAME_HEADER_FLOATS * 4).unwrap();
        let event_size = i32::try_from(EVENT_FLOATS * 4).unwrap();
        let event = |offset| [load(0), load(24), vec![0x6a, 0x2a, 2, offset]].concat();
        main.extend(store(lane(receipt + ui_node::VALUE), integer((-1.0_f32).to_bits().cast_signed())));
        main.extend(store(24, integer(first_event)));
        main.extend(store(
            28,
            [request(24), integer(event_size), vec![0x6c], integer(first_event), vec![0x6a]].concat(),
        ));
        main.extend([0x02, 0x40, 0x03, 0x40]); // block; loop
        main.extend([load(24), load(28), vec![0x4f, 0x0d, 1]].concat()); // exit when offset >= end
        main.extend([event(0), float(event_tags::UI_ACTION as f32), vec![0x5b, 0x04, 0x40]].concat());
        main.extend(
            [
                integer(lane(receipt + ui_node::VALUE)),
                event(8),
                float(100.0),
                vec![0x94], // node * 100
                event(4),
                float(10.0),
                vec![0x94, 0x92], // + action * 10
                event(12),
                vec![0x92, 0x38, 2, 0], // + value; f32.store
                integer(lane(unnamed + ui_node::HIDDEN)),
                event(4),
                float(ui_action::BLUR as f32),
                vec![0x5c, 0xb2, 0x38, 2, 0, 0x0b], // f32.ne; f32.convert_i32_s; f32.store; end if
            ]
            .concat(),
        );
        main.extend(store(24, [load(24), integer(event_size), vec![0x6a]].concat()));
        main.extend([0x0c, 0, 0x0b, 0x0b]); // next event; end loop and block
        main.extend([store(20, integer(RENDER)), exchange(), vec![0x0c, 0, 0x0b], integer(0), vec![0x0b]].concat());
        program(
            main,
            &[
                (HANDSHAKE, &[4670544.0, 2.0, 1.0, 0.0]),
                (CREATE, &[1.0, 40.0]),
                (SNAPSHOT, &[3.0, 0.0]),
                (RESTORE, &[4.0, 0.0]),
                (RENDER, &render),
            ],
        )
    }

    fn start(directory: &Path) -> Session {
        let assets = directory.join("assets");
        std::fs::create_dir_all(&assets).unwrap();
        let engine = squid_host::engine().unwrap();
        let gpu = Gpu::headless((64, 48), &assets, false).unwrap();
        let module = GuestModule::new(&engine, &fixture(), None).unwrap();
        Session::start(
            Options {
                frames: 0,
                screenshot: None,
                record: None,
                fps: Some(60.0),
                mute: true,
                audio_buffer: None,
                dev: false,
                script: Vec::new(),
                headless: true,
                settle: false,
            },
            Paths::new(directory.to_owned(), Some(assets), None, None, Some(directory.join("data"))),
            engine,
            module,
            Catalog { meshes: vec![], materials: vec![], sounds: vec![] },
            gpu,
            None,
        )
        .unwrap()
    }

    fn assert_queries(tester: &SessionTester<'_>) {
        let screen = tester.screen();
        let slider = screen.get_by_role_and_name("SLIDER", "Music Volume").unwrap();
        assert_eq!(slider.id, SLIDER);
        assert_eq!(screen.by_role("slider").name("music volume").query().unwrap(), None);
        assert_eq!(screen.by_role("slider").name("music volume").case_insensitive().get().unwrap(), slider);
        assert_eq!(
            screen
                .by_role("slider")
                .name_contains("MUSIC")
                .case_insensitive()
                .description("Audio")
                .value_text("Half")
                .expanded(false)
                .disabled(false)
                .focused(true)
                .get()
                .unwrap(),
            slider
        );
        assert_eq!(screen.get_by_label_text("Music Volume").unwrap(), slider);
        let absent = screen.get_by_role("tab").unwrap_err();
        assert!(matches!(absent, QueryError::NotFound { .. }));
        assert!(absent.to_string().contains("Music Volume") && absent.to_string().contains("<- focus"));
        assert_eq!(screen.query_by_role("tab").unwrap(), None);
        let ambiguous = screen.get_by_role("checkbox").unwrap_err();
        assert!(matches!(ambiguous, QueryError::MultipleMatches { count: 3, .. }));
        assert!(ambiguous.to_string().contains("active modal #10"));
        assert_eq!(screen.query_by_role("checkbox").unwrap_err(), ambiguous);
        assert_eq!(screen.by_role("checkbox").checked(true).get().unwrap().name, "All");
        assert_eq!(screen.by_role("checkbox").checked(false).get().unwrap().name, "None");
        assert_eq!(screen.by_role("checkbox").checked_state(ui_state::MIXED).get().unwrap().name, "Some");
        let list = screen.get_by_role("listbox").unwrap();
        assert_eq!(screen.get_by_role("list").unwrap(), list);
        let option = screen.within_node(&list).by_role("option").selected(true).get().unwrap();
        assert_eq!(option.id, OPTION);
        assert_eq!(screen.within(list.id).get_by_role("listitem").unwrap(), option);
        assert!(screen.within(list.id).query_by_role("slider").unwrap().is_none());
        assert_eq!(screen.by_role("button").disabled(true).get().unwrap().id, DISABLED);
        assert_eq!(screen.by_role("button").name("Outside").query().unwrap(), None);
        let outside = screen.by_role("button").name("Outside").include_hidden(true).get().unwrap();
        assert!(outside.blocked_by_modal);
        assert_eq!(screen.click(&outside), Err(ActionError::BlockedByModal { id: OUTSIDE }));
    }

    #[test]
    fn queries_and_dispatches_to_a_live_headless_session() {
        let directory = Path::new("/var/tmp").join(format!("pen-role-testing-{}", std::process::id()));
        let mut session = start(&directory);
        let mut tester = SessionTester::new(&mut session);
        assert_queries(&tester);
        tester.assert_audit_clean().unwrap();

        assert!(tester.step().unwrap());
        let first = tester.session.timeline().frame;
        let assert_receipt = |tester: &SessionTester<'_>, value: f32| {
            assert_eq!(tester.screen().get_by_role("progressbar").unwrap().value, value);
        };
        assert_receipt(&tester, -1.0);

        let blocked = tester.click_by_role("button", "Outside").unwrap_err();
        assert!(matches!(blocked.downcast_ref::<QueryError>(), Some(QueryError::NotFound { .. })));
        for (result, expected) in [
            (tester.click_by_role("button", "Apply"), ActionError::Disabled { id: DISABLED }),
            (tester.click_by_role("checkbox", "All"), ActionError::Unsupported { id: 14, action: ui_action::ACTIVATE }),
            (tester.set_value_by_role("slider", "Music Volume", f32::NAN), ActionError::NonFiniteValue { id: SLIDER }),
        ] {
            assert_eq!(result.unwrap_err().downcast_ref(), Some(&expected));
        }
        assert_eq!(tester.session.timeline().frame, first);

        let pre_action = tester.click_by_role("option", "Low").unwrap();
        assert_eq!(pre_action.id, OPTION);
        assert_receipt(&tester, 1320.0);
        assert_eq!(tester.session.timeline().frame, first + 1);
        let pre_action = tester.focus_by_role(ui_role::SLIDER, "Music Volume").unwrap();
        assert_eq!(pre_action.id, SLIDER);
        assert_receipt(&tester, 1100.0);
        tester.increment_by_role("slider", "Music Volume").unwrap();
        assert_receipt(&tester, 1130.0);
        tester.decrement_by_role("slider", "Music Volume").unwrap();
        assert_receipt(&tester, 1140.0);
        let pre_action = tester.set_value_by_role("slider", "Music Volume", 0.75).unwrap();
        assert_eq!(pre_action.value, 0.5);
        assert_receipt(&tester, 1150.75);
        tester.expand_by_role("slider", "Music Volume").unwrap();
        assert_receipt(&tester, 1160.0);
        tester.collapse_by_role("slider", "Music Volume").unwrap();
        assert_receipt(&tester, 1170.0);
        assert_eq!(tester.session.timeline().frame, first + 7);

        let screen = tester.screen();
        let slider = screen.get_by_role("slider").unwrap();
        let event = screen.blur(&slider).unwrap();
        assert!(tester.dispatch(event).unwrap());
        assert_receipt(&tester, 1110.0);
        assert_eq!(tester.session.timeline().frame, first + 8);
        let failure = tester.assert_audit_clean().unwrap_err().to_string();
        assert!(failure.contains("button #0x16 has no name") && failure.contains("Accessible tree"));
        session.finish().unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }
}
