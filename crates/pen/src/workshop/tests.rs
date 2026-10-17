//! The Workshop's requests against a stand-in for Steam that records what
//! it was asked and calls back when the test says.
use std::collections::HashMap;

use pen_protocol::service_status;
use pen_runtime::services::code_points;

use super::*;
use crate::snapshot::{self, SavedState};

#[derive(Default)]
struct Fake {
    created: Vec<u32>,
    submitted: Vec<(u32, u64, Update)>,
    subscriptions: Vec<(u32, u64, bool)>,
    subscribed: Vec<u64>,
    installed: HashMap<u64, PathBuf>,
    downloads: Vec<u64>,
    done: Vec<Done>,
}

impl Ugc for Fake {
    fn create(&mut self, request: u32) {
        self.created.push(request);
    }
    fn submit(&mut self, request: u32, item: u64, update: &Update) {
        // Steam reads the content while it uploads: it is there now.
        assert!(update.content.join(SCENE).is_file());
        self.submitted.push((request, item, update.clone()));
    }
    fn subscribe(&mut self, request: u32, item: u64) {
        self.subscriptions.push((request, item, true));
    }
    fn unsubscribe(&mut self, request: u32, item: u64) {
        self.subscriptions.push((request, item, false));
    }
    fn subscribed(&mut self) -> Vec<u64> {
        self.subscribed.clone()
    }
    fn installed(&mut self, item: u64) -> Option<PathBuf> {
        self.installed.get(&item).cloned()
    }
    fn download(&mut self, item: u64) -> bool {
        self.downloads.push(item);
        true
    }
    fn finished(&mut self) -> Vec<Done> {
        std::mem::take(&mut self.done)
    }
}

fn directory(name: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!("pen-workshop-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

fn ask(workshop: &mut Workshop<Fake>, id: u32, operation: u32, numbers: [f32; 4], text: &str) -> Option<Answer> {
    let values = code_points(text);
    workshop.call(&Request { id, operation, numbers, values: &values })
}

fn text(answer: &Answer) -> String {
    pen_runtime::services::text(&answer.values)
}

#[test]
fn the_players_scene_is_published_once_steam_has_the_item() {
    let data = directory("publish");
    let scene = data.join("main.scene");
    snapshot::write(&scene, &SavedState { schema: 26, state: vec![1.0, 2.0] }).unwrap();
    let mut workshop = Workshop::new(Fake::default(), scene, staging(&data.join("main.scene")));
    // A new item, friends only: answered once Steam has created and taken it.
    assert_eq!(ask(&mut workshop, 7, steam_op::WORKSHOP_PUBLISH, [1.0, 0.0, 0.0, 0.0], "My meadow"), None);
    assert_eq!(workshop.ugc.created, [7]);
    assert!(workshop.poll().is_empty());
    workshop.ugc.done.push(Done::Created { request: 7, result: Ok((3141592653589, true)) });
    assert!(workshop.poll().is_empty(), "uploading");
    let (request, item, update) = workshop.ugc.submitted[0].clone();
    assert_eq!(
        (request, item, update.title.as_str(), update.visibility),
        (7, 3141592653589, "My meadow", Visibility::Friends)
    );
    workshop.ugc.done.push(Done::Submitted { request: 7, result: Ok((3141592653589, true)) });
    let answers = workshop.poll();
    assert_eq!(answers.len(), 1);
    let (request, answer) = &answers[0];
    // Its id, as text (it does not fit a float), and that the player must
    // accept the Workshop's agreement.
    assert_eq!(
        (*request, answer.status, answer.value, text(answer).as_str()),
        (7, service_status::OK, 1.0, "3141592653589")
    );
    assert!(!update.content.exists(), "the staged upload goes once Steam has it");

    // A new version of it, then a failed one.
    assert_eq!(
        ask(&mut workshop, 8, steam_op::WORKSHOP_PUBLISH, [0.0, 1.0, 0.0, 0.0], "3141592653589\nMeadow 2"),
        None
    );
    assert_eq!(workshop.ugc.submitted[1].1, 3141592653589);
    assert_eq!(workshop.ugc.submitted[1].2.title, "Meadow 2");
    workshop.ugc.done.push(Done::Submitted { request: 8, result: Err("Steam is busy".into()) });
    assert_eq!(workshop.poll()[0].1.status, service_status::FAILED);
    std::fs::remove_dir_all(data).unwrap();
}

#[test]
fn what_steam_would_refuse_is_refused_first() {
    let data = directory("refuse");
    let scene = data.join("main.scene");
    let mut workshop = Workshop::new(Fake::default(), scene.clone(), staging(&scene));
    // Nothing saved in the editor yet.
    let answer = ask(&mut workshop, 1, steam_op::WORKSHOP_PUBLISH, [0.0; 4], "Meadow").unwrap();
    assert_eq!(answer.status, service_status::FAILED);
    snapshot::write(&scene, &SavedState { schema: 26, state: vec![1.0] }).unwrap();
    for (numbers, title) in [([0.0; 4], ""), ([0.0; 4], &*"x".repeat(129)), ([9.0, 0.0, 0.0, 0.0], "Meadow")] {
        let answer = ask(&mut workshop, 2, steam_op::WORKSHOP_PUBLISH, numbers, title).unwrap();
        assert_eq!(answer.status, service_status::FAILED, "{title:?}");
    }
    assert!(workshop.ugc.created.is_empty());
    let content = data.join("content");
    std::fs::create_dir_all(&content).unwrap();
    let update = |preview: Option<PathBuf>| Update {
        title: "a\0b".into(),
        description: Some("x".repeat(8001)),
        content: content.clone(),
        preview,
        tags: vec![" ".into()],
        visibility: Visibility::Public,
        note: None,
    };
    assert!(update(None).check().is_err(), "an empty content folder");
    std::fs::write(content.join(SCENE), b"scene").unwrap();
    assert!(update(None).check().is_err(), "a description past 8000 characters");
    let fine = Update { description: None, ..update(Some(data.join("preview.bmp"))) };
    assert!(fine.clone().check().is_err(), "a preview that is no image");
    std::fs::write(data.join("preview.png"), vec![0; 2 << 20]).unwrap();
    assert!(Update { preview: Some(data.join("preview.png")), ..fine.clone() }.check().is_err(), "past 1 MB");
    let checked = Update { preview: None, ..fine }.check().unwrap();
    assert_eq!((checked.title.as_str(), checked.tags.len()), ("ab", 0));
    std::fs::remove_dir_all(data).unwrap();
}

#[test]
fn subscribed_items_install_and_become_the_editors_scene() {
    let data = directory("use");
    let scene = data.join("player/main.scene");
    let mut workshop = Workshop::new(Fake::default(), scene.clone(), staging(&scene));
    let folder = data.join("item");
    std::fs::create_dir_all(&folder).unwrap();
    let theirs = SavedState { schema: 26, state: vec![4.0, 5.0] };
    snapshot::write(&folder.join(SCENE), &theirs).unwrap();
    workshop.ugc.subscribed = vec![11, 12];
    workshop.ugc.installed.insert(11, folder);

    // Subscribing is answered when Steam calls back.
    assert_eq!(ask(&mut workshop, 1, steam_op::WORKSHOP_SUBSCRIBE, [0.0; 4], "12"), None);
    assert_eq!(ask(&mut workshop, 2, steam_op::WORKSHOP_UNSUBSCRIBE, [0.0; 4], "13"), None);
    assert_eq!(workshop.ugc.subscriptions, [(1, 12, true), (2, 13, false)]);
    workshop.ugc.done.push(Done::Subscribed { request: 1, result: Ok(()) });
    workshop.ugc.done.push(Done::Unsubscribed { request: 2, result: Err("no such item".into()) });
    let statuses: Vec<u32> = workshop.poll().iter().map(|(_, answer)| answer.status).collect();
    assert_eq!(statuses, [service_status::OK, service_status::FAILED]);
    assert_eq!(
        ask(&mut workshop, 3, steam_op::WORKSHOP_SUBSCRIBE, [0.0; 4], "twelve").unwrap().status,
        service_status::FAILED
    );

    // The installed ones, and the rest downloading.
    let items = ask(&mut workshop, 4, steam_op::WORKSHOP_ITEMS, [0.0; 4], "").unwrap();
    assert_eq!((items.value, text(&items).as_str()), (1.0, "11"));
    assert_eq!(workshop.ugc.downloads, [12]);

    // An installed item's scene becomes the player's; one not installed yet
    // is not, and downloads.
    let used = ask(&mut workshop, 5, steam_op::WORKSHOP_USE, [0.0; 4], "11").unwrap();
    assert_eq!(used.status, service_status::OK);
    assert_eq!(snapshot::read(&scene).unwrap(), theirs);
    assert_eq!(ask(&mut workshop, 6, steam_op::WORKSHOP_USE, [0.0; 4], "12").unwrap().status, service_status::FAILED);
    assert_eq!(workshop.ugc.downloads, [12, 12]);
    std::fs::remove_dir_all(data).unwrap();
}
