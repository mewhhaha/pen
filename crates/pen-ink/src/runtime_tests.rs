#[cfg(test)]
mod tests {
    use pen_protocol::boundary::{PACKET_HEADER_FLOATS, PACKET_MAGIC, lanes};

    use crate::saved::STATE_MAGIC;
    use crate::{Ink, SavedTypes, StateAdapter};
    use anyhow::Result;
    use pen_protocol::boundary::whole;
    use pen_protocol::{Catalog, RenderFrame, Viewport};
    use pen_runtime::game::{GameRuntime, Host, ReloadResult};
    use pen_runtime::program::{Backend, Operation, Program};
    use pen_runtime::program::{Response, response};
    use std::sync::Arc;
    use web_time::Instant;

    /// A game saved as ink's persist saves one: the entities, and section
    /// 100, `points` (version 1) or `points, bonus` (version 2, which has no
    /// migration from 1). Each module is a build: its version.
    #[derive(Clone)]
    struct Builds;

    struct Persisting {
        version: u32,
        state: Vec<f32>,
        answer: Option<(Operation, Vec<f32>)>,
    }

    fn state(version: u32, record: &[f32]) -> Vec<f32> {
        let mut state = vec![STATE_MAGIC, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        state.extend([100.0, version as f32, 0.0, record.len() as f32]);
        state.extend_from_slice(record);
        state
    }

    impl Backend for Builds {
        type Module = u32;
        type Program = Persisting;
        async fn start(&self, version: &u32) -> Result<Persisting> {
            Ok(Persisting { version: *version, state: Vec::new(), answer: None })
        }
    }

    impl Program for Persisting {
        fn schema(&self) -> u32 {
            1
        }
        fn scenes(&self) -> bool {
            false
        }
        fn send(&mut self, operation: Operation, payload: &[f32]) -> Result<()> {
            let lanes = match operation {
                Operation::Create => {
                    self.state = if self.version == 1 { state(1, &[42.0]) } else { state(2, &[0.0, 0.25]) };
                    Vec::new()
                }
                Operation::Snapshot => self.state.clone(),
                Operation::Restore => {
                    let saved = whole(payload[9]).unwrap();
                    let fate = if saved == self.version { 0.0 } else { 2.0 };
                    self.state = if fate == 0.0 { payload.to_vec() } else { state(2, &[0.0, 0.25]) };
                    vec![5680.0, 2.0, 0.0, 0.0, 1.0, 1.0, 100.0, fate, saved as f32, self.version as f32]
                }
                _ => {
                    let mut packet = vec![0.0; PACKET_HEADER_FLOATS];
                    packet[lanes::header::MAGIC] = PACKET_MAGIC;
                    packet[lanes::header::SCHEMA] = 1.0;
                    packet
                }
            };
            self.answer = Some((operation, [&[operation as u8 as f32][..], &lanes].concat()));
            Ok(())
        }
        async fn response(&mut self) -> Result<Response> {
            let (operation, answer) = self.answer.take().unwrap();
            response(operation, answer)
        }
        fn accept(&mut self) {}
        fn answered(&self) -> Option<Instant> {
            None
        }
        fn memory_bytes(&self) -> usize {
            0
        }
        async fn close(&mut self) {}
    }

    struct Viewer {
        declared: Option<SavedTypes>,
        opaque: bool,
        reports: Vec<String>,
    }

    impl Host for Viewer {
        fn viewport(&self) -> Viewport {
            Viewport { width: 64, height: 64 }
        }
        fn validate(&self, _: &RenderFrame) -> Result<()> {
            Ok(())
        }
        fn render(&mut self, _: &RenderFrame, _: &Catalog) -> Result<()> {
            Ok(())
        }
        fn report(&mut self, message: &str) {
            self.reports.push(message.to_owned());
        }
        fn state_adapter(&self) -> Arc<dyn StateAdapter> {
            if self.opaque {
                pen_runtime::state::opaque()
            } else {
                Arc::new(Ink { types: self.declared.clone().unwrap_or_default() })
            }
        }
    }

    fn ready<T>(future: impl Future<Output = T>) -> T {
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match std::pin::pin!(future).poll(&mut context) {
            std::task::Poll::Ready(value) => value,
            std::task::Poll::Pending => unreachable!("these programs answer at once"),
        }
    }

    const DECLARED: &str = r#"{"types": {"score": {"key": 100, "kind": "resource", "versions": {
        "1": [{"name": "points", "type": "f32"}],
        "2": [{"name": "points", "type": "f32"}, {"name": "bonus", "type": "f32", "default": 0.5}]}}}}"#;

    fn reload(declared: Option<SavedTypes>) -> (Viewer, Vec<f32>) {
        let catalog = Catalog { meshes: vec!["m".into()], materials: vec!["x".into()], sounds: Vec::new() };
        let mut viewer = Viewer { declared, opaque: false, reports: Vec::new() };
        let builds = Builds;
        let mut runtime = ready(GameRuntime::new(&builds, 1, catalog.clone(), &mut viewer, None, None)).unwrap();
        assert_eq!(ready(runtime.reload(&mut viewer, 2, catalog)), ReloadResult::Activated);
        let saved = ready(runtime.snapshot()).unwrap();
        (viewer, saved.state)
    }

    #[test]
    fn reloads_migrate_what_the_game_cannot_as_declared() {
        let (viewer, state) = reload(Some(SavedTypes::parse(DECLARED).unwrap()));
        assert_eq!(state[8..], [100.0, 2.0, 0.0, 2.0, 42.0, 0.5], "the points kept, the bonus its default");
        let reported = viewer.reports.last().unwrap();
        assert!(reported.contains("migrated section 100 from version 1 to 2 as saved.json declares"), "{reported}");
        // Without the declaration the game resets the section.
        let (viewer, state) = reload(None);
        assert_eq!(state[8..], [100.0, 2.0, 0.0, 2.0, 0.0, 0.25]);
        assert!(viewer.reports.last().unwrap().contains("reset section 100"), "{:?}", viewer.reports);
    }
    #[test]
    fn opaque_runtime_round_trips_payload_without_interpreting_restore_response() {
        let catalog = Catalog { meshes: vec![], materials: vec![], sounds: vec![] };
        let mut viewer = Viewer { declared: Some(SavedTypes::parse(DECLARED).unwrap()), opaque: true, reports: vec![] };
        let mut runtime = ready(GameRuntime::new(&Builds, 1, catalog.clone(), &mut viewer, None, None)).unwrap();
        let saved = ready(runtime.snapshot()).unwrap();
        ready(runtime.load(&mut viewer, &saved)).unwrap();
        assert_eq!(ready(runtime.snapshot()).unwrap(), saved);
        assert_eq!(ready(runtime.reload(&mut viewer, 2, catalog)), ReloadResult::Activated);
        // Even with declarations present, opaque state never invokes the Ink migration.
        assert_eq!(ready(runtime.snapshot()).unwrap().state[8..], [100.0, 2.0, 0.0, 2.0, 0.0, 0.25]);
        assert!(viewer.reports.iter().all(|report| !report.contains("section")), "{:?}", viewer.reports);
    }
}
