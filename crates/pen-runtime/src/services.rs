//! Services: what the platform and native libraries do for the game, through
//! one deterministic boundary (gdev's `packages/services.blot`). The game asks
//! in its render packet's `services` section; a [`Service`] answers each
//! request once, and the answer reaches the game as input, a ServiceAnswer
//! event and then ServiceData events carrying its values three at a time, in
//! the next frame that runs live. Answers are input, so a recording of the
//! input holds them and a replay, rewind or step through recorded frames
//! plays them back without calling a service again: hosts call services for
//! live frames only.
//!
//! The registry ([`Services`]) has one service per code
//! (`pen_protocol::service`); a request for a code nobody registered is
//! answered `unknown`, and one a service cannot do here `unavailable`. The
//! answers both hosts give alike live here: grid pathfinding, what the game
//! runs on ([`Platform`]) and which web addresses may be opened; the
//! simulation services (physics, navmesh, spatial queries and noise) are
//! pen-simulation's.
//!
//! A service answers at once, or later: [`Service::call`] returns None and
//! [`Service::poll`] hands the answer over when it is ready (a dialog
//! closing, a browser promise settling). Answers wait in the registry until
//! the next live frame, and a restore (a load, undo, reload or time travel)
//! drops those waiting: they were for the world it replaced.
use pen_protocol::boundary::{MAX_SERVICE_DATA, whole};
use pen_protocol::{
    Catalog, GameEvent, RenderFrame, ServiceRequest, pathfinding_op, platform_op, service, service_status,
};

/// A request as a service sees it.
pub struct Request<'a> {
    pub id: u32,
    pub operation: u32,
    /// The request's four numbers, `a` to `d`.
    pub numbers: [f32; 4],
    /// Its values: text as code points, or a compute service's numbers.
    pub values: &'a [f32],
}

impl Request<'_> {
    /// The values as text; values that are no code point become U+FFFD.
    pub fn text(&self) -> String {
        text(self.values)
    }
}

impl<'a> From<&'a ServiceRequest> for Request<'a> {
    fn from(request: &'a ServiceRequest) -> Request<'a> {
        Request {
            id: request.request,
            operation: request.operation,
            numbers: [request.a, request.b, request.c, request.d],
            values: &request.values,
        }
    }
}

/// Code points as text; values that are no code point become U+FFFD.
pub fn text(values: &[f32]) -> String {
    values.iter().map(|&value| whole(value).and_then(char::from_u32).unwrap_or(char::REPLACEMENT_CHARACTER)).collect()
}

/// Text as code points.
pub fn code_points(text: &str) -> Vec<f32> {
    text.chars().map(|char| char as u32 as f32).collect()
}

/// A service's answer: how it went (`pen_protocol::service_status`), a
/// number, and values (text as code points, or numbers).
#[derive(Clone, Debug, PartialEq)]
pub struct Answer {
    pub status: u32,
    pub value: f32,
    pub values: Vec<f32>,
}

impl Answer {
    pub fn ok(value: f32) -> Answer {
        Answer { status: service_status::OK, value, values: Vec::new() }
    }

    pub fn text(text: &str) -> Answer {
        Answer { status: service_status::OK, value: 0.0, values: code_points(text) }
    }

    pub fn unavailable() -> Answer {
        Answer { status: service_status::UNAVAILABLE, value: 0.0, values: Vec::new() }
    }

    pub fn failed() -> Answer {
        Answer { status: service_status::FAILED, value: 0.0, values: Vec::new() }
    }

    pub fn unknown() -> Answer {
        Answer { status: service_status::UNKNOWN, value: 0.0, values: Vec::new() }
    }

    /// The events that carry it to the game: a ServiceAnswer, then its
    /// values three to a ServiceData, the last padded with zeros.
    pub fn events(&self, request: u32) -> Vec<GameEvent> {
        let values = &self.values[..self.values.len().min(MAX_SERVICE_DATA)];
        let value = if self.value.is_finite() { self.value } else { 0.0 };
        let length = u32::try_from(values.len()).expect("at most MAX_SERVICE_DATA values");
        let mut events = vec![GameEvent::ServiceAnswer { request, status: self.status, value, length }];
        for chunk in values.chunks(3) {
            let at = |index: usize| chunk.get(index).copied().filter(|value| value.is_finite()).unwrap_or(0.0);
            events.push(GameEvent::ServiceData { request, a: at(0), b: at(1), c: at(2) });
        }
        events
    }
}

/// A service the game can ask: the platform's, or a native library's.
/// `Input` is what the host gives services once a frame (native pen's input
/// queue, which Steam tells of its overlay).
pub trait Service<Input = ()> {
    /// The service code it answers (`pen_protocol::service`).
    fn code(&self) -> u32;

    /// Answers a request now, or returns None and answers it later through
    /// `poll`. Called for live frames only, once per request.
    fn call(&mut self, request: &Request) -> Option<Answer>;

    /// Answers that came in since the last frame, by request id.
    fn poll(&mut self) -> Vec<(u32, Answer)> {
        Vec::new()
    }

    /// Once a frame, before the game's.
    fn frame(&mut self, _input: &mut Input) {}

    /// A frame the host shows (live or replayed), with the catalog it
    /// names: what a service follows of what plays (captions).
    fn rendered(&mut self, _frame: &RenderFrame, _catalog: &Catalog) {}

    /// The game's world was replaced (a load, undo, reload or time travel).
    fn restored(&mut self) {}
}

/// The services a host answers with, and the answers waiting for the next
/// live frame.
pub struct Services<Input = ()> {
    services: Vec<Box<dyn Service<Input>>>,
    answers: Vec<GameEvent>,
    /// Requests answered (or taken to answer later) so far.
    calls: u64,
    /// A line on stderr for each request (`PEN_SERVICES_LOG`).
    log: bool,
}

impl<Input> Default for Services<Input> {
    fn default() -> Services<Input> {
        Services { services: Vec::new(), answers: Vec::new(), calls: 0, log: false }
    }
}

/// What a request was, for the log: `steam unlock_achievement "FIRST_BRICK"`.
fn describe(request: &ServiceRequest) -> String {
    let name = |names: &[&str], code: u32| names.get(code as usize).map_or(code.to_string(), |name| (*name).to_owned());
    let operations: &[&str] = match request.service {
        service::CAPTIONS => &pen_protocol::captions_op::NAMES,
        service::STEAM => &pen_protocol::steam_op::NAMES,
        service::CLIPBOARD => &pen_protocol::clipboard_op::NAMES,
        service::URL => &pen_protocol::url_op::NAMES,
        service::PLATFORM => &pen_protocol::platform_op::NAMES,
        service::PATHFINDING => &pen_protocol::pathfinding_op::NAMES,
        service::PHYSICS => &pen_protocol::physics_op::NAMES,
        service::NAVMESH => &pen_protocol::navmesh_op::NAMES,
        service::SPATIAL => &pen_protocol::spatial_op::NAMES,
        service::NOISE => &pen_protocol::noise_op::NAMES,
        _ => &[],
    };
    let shown = if request.service == service::PATHFINDING {
        let blocked = request.values.iter().filter(|value| **value != 0.0).count();
        format!("{} cells, {blocked} blocked", request.values.len())
    } else if matches!(request.service, service::PHYSICS | service::NAVMESH | service::SPATIAL | service::NOISE) {
        format!("{} values", request.values.len())
    } else {
        let text = text(&request.values);
        let text =
            if text.chars().count() > 40 { format!("{}…", text.chars().take(40).collect::<String>()) } else { text };
        format!("{text:?}")
    };
    format!(
        "{} {} {shown} [{}, {}, {}, {}]",
        name(&service::NAMES, request.service),
        name(operations, request.operation),
        request.a,
        request.b,
        request.c,
        request.d
    )
}

impl<Input> Services<Input> {
    /// None registered: every request is answered `unknown`.
    pub fn new() -> Services<Input> {
        Services { log: std::env::var_os("PEN_SERVICES_LOG").is_some(), ..Services::default() }
    }

    /// Adds a service, in place of any with the same code.
    pub fn register(&mut self, added: Box<dyn Service<Input>>) {
        self.services.retain(|registered| registered.code() != added.code());
        self.services.push(added);
    }

    /// Requests answered so far.
    pub fn calls(&self) -> u64 {
        self.calls
    }

    /// Once a frame: the services' own work, and answers that came in.
    pub fn frame(&mut self, input: &mut Input) {
        for registered in &mut self.services {
            registered.frame(input);
            for (id, answer) in registered.poll() {
                if self.log {
                    eprintln!("service: request {id} answered {}", service_status::NAMES[answer.status as usize]);
                }
                self.answers.extend(answer.events(id));
            }
        }
    }

    /// Asks the services a live frame's requests; their answers go to the
    /// game with the next live frame.
    pub fn answer(&mut self, requests: &[ServiceRequest]) {
        for request in requests {
            self.calls += 1;
            let asked = Request::from(request);
            let answer = match self.services.iter_mut().find(|registered| registered.code() == request.service) {
                Some(registered) => registered.call(&asked),
                None => Some(Answer::unknown()),
            };
            if self.log {
                let outcome = answer.as_ref().map_or("later".to_owned(), |answer| {
                    format!(
                        "{} {} ({} values)",
                        service_status::NAMES[answer.status as usize],
                        answer.value,
                        answer.values.len()
                    )
                });
                eprintln!("service: request {} {} -> {outcome}", asked.id, describe(request));
            }
            if let Some(answer) = answer {
                self.answers.extend(answer.events(asked.id));
            }
        }
    }

    /// The answers for the next live frame, in the order they came.
    pub fn take_answers(&mut self) -> Vec<GameEvent> {
        std::mem::take(&mut self.answers)
    }

    /// The game's world was replaced (a load, undo, reload or time travel):
    /// the answers for the one that asked are dropped.
    pub fn forget(&mut self) {
        self.answers.clear();
        for registered in &mut self.services {
            registered.restored();
        }
    }

    /// A frame the host shows, for the services that follow what plays.
    pub fn rendered(&mut self, frame: &RenderFrame, catalog: &Catalog) {
        for registered in &mut self.services {
            registered.rendered(frame, catalog);
        }
    }

    /// How many requests a session answered, once it ends.
    pub fn report(&self) {
        if self.calls() > 0 {
            eprintln!("services: answered {} requests", self.calls());
        }
    }
}

/// Compute as a service: the shortest 4-connected path over a grid, by A*.
/// The game sends the grid (`a` columns by `b` rows of cells, 0 open and
/// anything else blocked) and the start and goal cells (`c` and `d`, row *
/// columns + column); the answer's values are the path's cells from start to
/// goal and its value their count, or none when the goal cannot be reached.
/// Ties break the same way every time, so the same grid gives the same path
/// on every machine.
///
/// It is the example of a library behind the services boundary: the game
/// hands over plain numbers and gets plain numbers back, and a replay reads
/// the path from the recording instead of searching again.
pub struct Pathfinding;

impl<Input> Service<Input> for Pathfinding {
    fn code(&self) -> u32 {
        service::PATHFINDING
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        if request.operation != pathfinding_op::FIND {
            return Some(Answer::unknown());
        }
        let [columns, rows, start, goal] = request.numbers.map(|lane| whole(lane).map(|value| value as usize));
        let (Some(columns), Some(rows), Some(start), Some(goal)) = (columns, rows, start, goal) else {
            return Some(Answer::failed());
        };
        if columns == 0 || columns.checked_mul(rows) != Some(request.values.len()) {
            return Some(Answer::failed());
        }
        let open: Vec<bool> = request.values.iter().map(|cell| *cell == 0.0).collect();
        Some(match crate::pathfinding::find(columns, &open, start, goal) {
            Some(path) => Answer {
                value: path.len() as f32,
                values: path.into_iter().map(|cell| cell as f32).collect(),
                ..Answer::ok(0.0)
            },
            None => Answer::ok(0.0),
        })
    }
}

/// What the game runs on: the operating system, the player's locale (a BCP
/// 47 tag, none when unknown) and the device, as text the game can show or
/// act on. The device's value is 1 on a handheld (a Steam Deck).
pub struct Platform {
    pub os: String,
    pub locale: Option<String>,
    pub device: String,
    pub handheld: bool,
}

impl<Input> Service<Input> for Platform {
    fn code(&self) -> u32 {
        service::PLATFORM
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        Some(match request.operation {
            platform_op::OS => Answer::text(&self.os),
            platform_op::LOCALE => match &self.locale {
                Some(tag) => Answer::text(tag),
                None => Answer::unavailable(),
            },
            platform_op::DEVICE => Answer { value: f32::from(u8::from(self.handheld)), ..Answer::text(&self.device) },
            _ => Answer::unknown(),
        })
    }
}

/// A service the host does not have (Steam in a browser): every request is
/// answered `unavailable`.
pub struct Unavailable(pub u32);

impl<Input> Service<Input> for Unavailable {
    fn code(&self) -> u32 {
        self.0
    }

    fn call(&mut self, _request: &Request) -> Option<Answer> {
        Some(Answer::unavailable())
    }
}

/// Whether `address` is a web address the game may have opened: http or
/// https, without spaces, control characters or quotes.
pub fn openable(address: &str) -> bool {
    (address.starts_with("https://") || address.starts_with("http://"))
        && address.len() <= 2048
        && address.chars().all(|char| !char.is_whitespace() && !char.is_control() && char != '"')
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts its calls and answers each with its id.
    struct Counting {
        calls: std::rc::Rc<std::cell::Cell<u32>>,
    }

    impl Service for Counting {
        fn code(&self) -> u32 {
            service::PLATFORM
        }

        fn call(&mut self, request: &Request) -> Option<Answer> {
            self.calls.set(self.calls.get() + 1);
            Some(Answer { status: service_status::OK, value: request.id as f32, values: code_points("linux") })
        }
    }

    fn request(id: u32, service: u32, values: Vec<f32>) -> ServiceRequest {
        ServiceRequest {
            request: id,
            service,
            value_count: u32::try_from(values.len()).unwrap(),
            values,
            ..ServiceRequest::default()
        }
    }

    #[test]
    fn answers_travel_as_events_three_values_at_a_time() {
        let events = Answer::text("linux").events(7);
        assert_eq!(
            events,
            [
                GameEvent::ServiceAnswer { request: 7, status: service_status::OK, value: 0.0, length: 5 },
                GameEvent::ServiceData { request: 7, a: 108.0, b: 105.0, c: 110.0 },
                GameEvent::ServiceData { request: 7, a: 117.0, b: 120.0, c: 0.0 },
            ]
        );
        assert_eq!(Answer::ok(2.5).events(1).len(), 1);
        assert_eq!(text(&code_points("Steam Deck é")), "Steam Deck é");
        assert_eq!(text(&[65.0, 0.5, -1.0]), "A\u{fffd}\u{fffd}");
    }

    #[test]
    fn requests_reach_their_service_once_and_unknown_ones_say_so() {
        let calls = std::rc::Rc::new(std::cell::Cell::new(0));
        let mut services: Services = Services::new();
        services.register(Box::new(Counting { calls: calls.clone() }));
        services.answer(&[request(3, service::PLATFORM, vec![]), request(4, 99, vec![])]);
        assert_eq!(calls.get(), 1);
        assert_eq!(services.calls(), 2);
        let answers = services.take_answers();
        assert_eq!(
            answers[0],
            GameEvent::ServiceAnswer { request: 3, status: service_status::OK, value: 3.0, length: 5 }
        );
        assert_eq!(
            answers.last(),
            Some(&GameEvent::ServiceAnswer { request: 4, status: service_status::UNKNOWN, value: 0.0, length: 0 })
        );
        assert!(services.take_answers().is_empty());
        // A restore drops what waits for the next frame.
        services.answer(&[request(5, service::PLATFORM, vec![])]);
        services.forget();
        assert!(services.take_answers().is_empty());
    }

    /// Answers on the frame after it was asked, as a dialog closing would.
    #[derive(Default)]
    struct Later(Vec<u32>, Vec<u32>);

    impl Service for Later {
        fn code(&self) -> u32 {
            service::CLIPBOARD
        }

        fn call(&mut self, request: &Request) -> Option<Answer> {
            self.0.push(request.id);
            None
        }

        fn poll(&mut self) -> Vec<(u32, Answer)> {
            self.1.drain(..).map(|id| (id, Answer::ok(1.0))).collect()
        }

        fn frame(&mut self, _input: &mut ()) {
            self.1.append(&mut self.0);
        }
    }

    #[test]
    fn answers_given_later_come_with_a_later_frame() {
        let mut services: Services = Services::new();
        services.register(Box::new(Later::default()));
        services.answer(&[request(8, service::CLIPBOARD, vec![])]);
        assert!(services.take_answers().is_empty());
        services.frame(&mut ());
        assert_eq!(services.take_answers(), Answer::ok(1.0).events(8));
    }

    #[test]
    fn pathfinding_answers_with_the_cells_or_fails_on_a_bad_grid() {
        let values: Vec<f32> = [0, 0, 1, 0, 0, 0].iter().map(|cell| *cell as f32).collect();
        let ask = |numbers: [f32; 4]| {
            Service::<()>::call(
                &mut Pathfinding,
                &Request { id: 1, operation: pathfinding_op::FIND, numbers, values: &values },
            )
            .unwrap()
        };
        let answer = ask([3.0, 2.0, 0.0, 5.0]);
        assert_eq!(answer.value, 4.0);
        assert_eq!(answer.values, [0.0, 1.0, 4.0, 5.0]);
        assert_eq!(ask([4.0, 2.0, 0.0, 5.0]).status, service_status::FAILED);
        assert_eq!(ask([3.0, 2.0, 0.5, 5.0]).status, service_status::FAILED);
    }

    #[test]
    fn the_platform_says_what_it_is_and_steam_elsewhere_is_unavailable() {
        let mut services: Services = Services::new();
        services.register(Box::new(Platform {
            os: "web".into(),
            locale: None,
            device: "browser".into(),
            handheld: false,
        }));
        services.register(Box::new(Unavailable(service::STEAM)));
        let ask = |operation| ServiceRequest { operation, ..request(1, service::PLATFORM, vec![]) };
        services.answer(&[ask(platform_op::OS), ask(platform_op::LOCALE), request(2, service::STEAM, vec![])]);
        let answers = services.take_answers();
        assert_eq!(answers[..2], Answer::text("web").events(1)[..2]);
        let statuses: Vec<u32> = answers
            .iter()
            .filter_map(|event| match event {
                GameEvent::ServiceAnswer { status, .. } => Some(*status),
                _ => None,
            })
            .collect();
        assert_eq!(statuses, [service_status::OK, service_status::UNAVAILABLE, service_status::UNAVAILABLE]);
    }

    #[test]
    fn only_web_addresses_open() {
        assert!(openable("https://example.com/gdev?from=game"));
        assert!(openable("http://localhost:8000/?fresh"));
        assert!(!openable("file:///etc/passwd"));
        assert!(!openable("https://example.com/a b"));
        assert!(!openable("javascript:alert(1)"));
        assert!(!openable("https://example.com/\"&calc"));
    }
}
