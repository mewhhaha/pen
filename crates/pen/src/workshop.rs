//! Steam Workshop: players publish the levels they build in the game's
//! editor and play the ones they subscribe to, through the Steam service's
//! workshop operations (`pen_protocol::steam_op`). Publishing uploads the
//! player's saved scene (what the editor saves) as a new item, or a new
//! version of one; `workshop_items` says which subscribed items are
//! installed, downloading the rest; and `workshop_use` puts an installed
//! item's scene where the editor loads it, so the game's own scene load
//! (recorded like any load) brings it in. Steam calls back later, so
//! publishing and subscribing are answered in a later frame (`poll`).
//!
//! The logic is generic over [`Ugc`], so it is tested without Steam;
//! `Steamworks` (the `steam` feature) is the Steam client's.
// Without the `steam` feature, only the tests use it.
#![cfg_attr(not(feature = "steam"), allow(dead_code))]
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use pen_protocol::steam_op;
use pen_runtime::services::{Answer, Request};

/// What Steam's Workshop takes at most (`k_cchPublishedDocumentTitleMax`,
/// `k_cchPublishedDocumentDescriptionMax`, and a preview image's size).
const MAX_TITLE: usize = 128;
const MAX_DESCRIPTION: usize = 8000;
const MAX_PREVIEW_BYTES: u64 = 1 << 20;
/// The scene an item carries, in its content folder.
pub const SCENE: &str = "main.scene";

/// Who may see a published item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Friends,
    Private,
    Unlisted,
}

impl Visibility {
    /// A request's `a`: 0 public, 1 friends, 2 private, 3 unlisted.
    fn from_code(code: f32) -> Option<Visibility> {
        Some(match pen_protocol::boundary::whole(code)? {
            0 => Visibility::Public,
            1 => Visibility::Friends,
            2 => Visibility::Private,
            3 => Visibility::Unlisted,
            _ => return None,
        })
    }
}

/// A new version of an item: what Steam is told and uploads.
#[derive(Clone, Debug, PartialEq)]
pub struct Update {
    pub title: String,
    pub description: Option<String>,
    /// The folder uploaded as the item's content.
    pub content: PathBuf,
    pub preview: Option<PathBuf>,
    pub tags: Vec<String>,
    pub visibility: Visibility,
    pub note: Option<String>,
}

impl Update {
    /// Refuses what Steam would: an empty or long title or description, an
    /// empty content folder, a preview that is no image or past 1 MB. Text
    /// loses the NULs Steam's strings cannot carry.
    pub fn check(mut self) -> Result<Update> {
        let clean = |text: &str| text.replace('\0', "").trim().to_owned();
        self.title = clean(&self.title);
        self.description = self.description.as_deref().map(clean);
        self.note = self.note.as_deref().map(clean);
        self.tags = self.tags.iter().map(|tag| clean(tag)).filter(|tag| !tag.is_empty()).collect();
        let length = self.title.chars().count();
        ensure!((1..=MAX_TITLE).contains(&length), "a Workshop title has 1 to {MAX_TITLE} characters, not {length}");
        ensure!(
            self.description.as_ref().is_none_or(|text| text.chars().count() <= MAX_DESCRIPTION),
            "a Workshop description has at most {MAX_DESCRIPTION} characters"
        );
        let empty = std::fs::read_dir(&self.content)
            .with_context(|| format!("reading the content folder {}", self.content.display()))?
            .next()
            .is_none();
        ensure!(!empty, "the content folder {} is empty", self.content.display());
        if let Some(preview) = &self.preview {
            let image = preview.extension().and_then(|extension| extension.to_str()).is_some_and(|extension| {
                ["png", "jpg", "jpeg", "gif"].contains(&extension.to_ascii_lowercase().as_str())
            });
            ensure!(image, "a Workshop preview is a PNG, JPEG or GIF, not {}", preview.display());
            let bytes = std::fs::metadata(preview).with_context(|| format!("reading {}", preview.display()))?.len();
            ensure!(bytes <= MAX_PREVIEW_BYTES, "a Workshop preview is at most 1 MB, not {bytes} bytes");
        }
        Ok(self)
    }
}

/// A call Steam finished, for the request that made it.
#[derive(Clone, Debug, PartialEq)]
pub enum Done {
    /// A new item, and whether the player must accept the Workshop's legal
    /// agreement before others see it.
    Created {
        request: u32,
        result: Result<(u64, bool), String>,
    },
    Submitted {
        request: u32,
        result: Result<(u64, bool), String>,
    },
    Subscribed {
        request: u32,
        result: Result<(), String>,
    },
    Unsubscribed {
        request: u32,
        result: Result<(), String>,
    },
}

/// What pen asks of Steam's UGC interface. Calls that Steam answers later
/// say so through `finished`.
pub trait Ugc {
    fn create(&mut self, request: u32);
    fn submit(&mut self, request: u32, item: u64, update: &Update);
    fn subscribe(&mut self, request: u32, item: u64);
    fn unsubscribe(&mut self, request: u32, item: u64);
    /// The items the player subscribes to.
    fn subscribed(&mut self) -> Vec<u64>;
    /// An installed item's folder.
    fn installed(&mut self, item: u64) -> Option<PathBuf>;
    /// Asks Steam to download an item; whether it will.
    fn download(&mut self, item: u64) -> bool;
    fn finished(&mut self) -> Vec<Done>;
}

/// An upload under way: the request asking, and its staged content.
struct Upload {
    request: u32,
    update: Update,
    /// The item, when it is a new version of one.
    item: Option<u64>,
}

pub struct Workshop<U> {
    ugc: U,
    /// The player's scene: what is published, and where an item's goes.
    scene: PathBuf,
    /// Where uploads are staged until Steam has read them.
    staging: PathBuf,
    uploads: Vec<Upload>,
    /// Answers that came in, for `poll`.
    answers: Vec<(u32, Answer)>,
}

/// Whether a Steam service `operation` is one of the Workshop's.
pub fn handles(operation: u32) -> bool {
    [
        steam_op::WORKSHOP_PUBLISH,
        steam_op::WORKSHOP_ITEMS,
        steam_op::WORKSHOP_SUBSCRIBE,
        steam_op::WORKSHOP_UNSUBSCRIBE,
        steam_op::WORKSHOP_USE,
    ]
    .contains(&operation)
}

/// An item's id, as the game sends it: its decimal digits.
fn item_id(text: &str) -> Result<u64> {
    let text = text.trim();
    ensure!(!text.is_empty() && text.chars().all(|digit| digit.is_ascii_digit()), "no Workshop item id: {text:?}");
    Ok(text.parse()?)
}

/// A failure, said on stderr and answered as failed.
fn failed(error: &anyhow::Error) -> Answer {
    eprintln!("Steam Workshop: {error:#}");
    Answer::failed()
}

fn ok_text(value: f32, text: &str) -> Answer {
    Answer { value, ..Answer::text(text) }
}

impl<U: Ugc> Workshop<U> {
    /// Publishes `scene`, the player's, and puts items' scenes there;
    /// stages uploads in `staging`.
    pub fn new(ugc: U, scene: PathBuf, staging: PathBuf) -> Workshop<U> {
        Workshop { ugc, scene, staging, uploads: Vec::new(), answers: Vec::new() }
    }

    /// Takes a request: its answer now, or None for one Steam gives later.
    pub fn call(&mut self, request: &Request) -> Option<Answer> {
        let result = match request.operation {
            steam_op::WORKSHOP_PUBLISH => self.publish(request).map(|()| None),
            steam_op::WORKSHOP_ITEMS => Ok(Some(self.items())),
            steam_op::WORKSHOP_SUBSCRIBE => item_id(&request.text()).map(|item| {
                self.ugc.subscribe(request.id, item);
                None
            }),
            steam_op::WORKSHOP_UNSUBSCRIBE => item_id(&request.text()).map(|item| {
                self.ugc.unsubscribe(request.id, item);
                None
            }),
            steam_op::WORKSHOP_USE => self.use_item(&request.text()).map(Some),
            _ => return Some(Answer::unknown()),
        };
        result.unwrap_or_else(|error| Some(failed(&error)))
    }

    /// `workshop_publish`: the player's scene as a new item titled with
    /// the text (`a` its visibility), or with `b` 1, a new version of the
    /// item whose id the text's first line is.
    fn publish(&mut self, request: &Request) -> Result<()> {
        let text = request.text();
        let (item, title) = match request.numbers[1] {
            1.0 => {
                let (id, title) =
                    text.split_once('\n').context("an update's text is the item's id, a line, the title")?;
                (Some(item_id(id)?), title.to_owned())
            }
            _ => (None, text),
        };
        let visibility = Visibility::from_code(request.numbers[0]).context("visibility is 0 to 3")?;
        let content = self.staging.join(format!("upload-{}", request.id));
        std::fs::create_dir_all(&content)?;
        std::fs::copy(&self.scene, content.join(SCENE))
            .with_context(|| format!("staging the scene {} (save it in the editor first)", self.scene.display()))?;
        let update = Update {
            title,
            description: None,
            content,
            preview: None,
            tags: vec!["scene".into()],
            visibility,
            note: None,
        }
        .check()?;
        match item {
            Some(item) => self.ugc.submit(request.id, item, &update),
            None => self.ugc.create(request.id),
        }
        self.uploads.push(Upload { request: request.id, update, item });
        Ok(())
    }

    /// `workshop_items`: the installed subscribed items' ids, a space
    /// between them, their count the value; the rest start downloading.
    fn items(&mut self) -> Answer {
        let mut installed = Vec::new();
        for item in self.ugc.subscribed() {
            if self.ugc.installed(item).is_some() {
                installed.push(item.to_string());
            } else {
                self.ugc.download(item);
            }
        }
        ok_text(installed.len() as f32, &installed.join(" "))
    }

    /// `workshop_use`: an installed item's scene becomes the player's, for
    /// the game to load as the editor's.
    fn use_item(&mut self, text: &str) -> Result<Answer> {
        let item = item_id(text)?;
        let Some(folder) = self.ugc.installed(item) else {
            self.ugc.download(item);
            bail!("item {item} is not installed yet (it is downloading)");
        };
        let scene = folder.join(SCENE);
        // Only a scene the game could load replaces the player's.
        crate::snapshot::read(&scene).with_context(|| format!("item {item} has no scene"))?;
        if let Some(parent) = self.scene.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(&scene, &self.scene)?;
        Ok(ok_text(0.0, &item.to_string()))
    }

    fn finish(&mut self, request: u32, answer: Answer) {
        if let Some(at) = self.uploads.iter().position(|upload| upload.request == request) {
            let upload = self.uploads.remove(at);
            let _ = std::fs::remove_dir_all(&upload.update.content);
        }
        self.answers.push((request, answer));
    }

    /// The answers Steam's calls brought since the last call.
    pub fn poll(&mut self) -> Vec<(u32, Answer)> {
        for done in self.ugc.finished() {
            match done {
                Done::Created { request, result: Ok((item, _)) } => {
                    match self.uploads.iter_mut().find(|upload| upload.request == request) {
                        Some(upload) => {
                            upload.item = Some(item);
                            let update = upload.update.clone();
                            self.ugc.submit(request, item, &update);
                        }
                        None => self.answers.push((request, Answer::failed())),
                    }
                }
                Done::Submitted { request, result: Ok((item, legal)) } => {
                    self.finish(request, ok_text(f32::from(u8::from(legal)), &item.to_string()));
                }
                Done::Subscribed { request, result: Ok(()) } | Done::Unsubscribed { request, result: Ok(()) } => {
                    self.answers.push((request, Answer::ok(0.0)));
                }
                Done::Created { request, result: Err(error) } | Done::Submitted { request, result: Err(error) } => {
                    let answer = failed(&anyhow::anyhow!("publishing failed: {error}"));
                    self.finish(request, answer);
                }
                Done::Subscribed { request, result: Err(error) }
                | Done::Unsubscribed { request, result: Err(error) } => {
                    self.answers.push((request, failed(&anyhow::anyhow!("{error}"))));
                }
            }
        }
        std::mem::take(&mut self.answers)
    }
}

/// The Steam client's Workshop.
#[cfg(feature = "steam")]
pub struct Steamworks {
    ugc: steamworks::UGC,
    app: steamworks::AppId,
    done: std::sync::Arc<std::sync::Mutex<Vec<Done>>>,
}

#[cfg(feature = "steam")]
impl Steamworks {
    pub fn new(client: &steamworks::Client) -> Steamworks {
        Steamworks { ugc: client.ugc(), app: client.utils().app_id(), done: Default::default() }
    }

    /// Where a call's result goes when Steam calls back.
    fn reply<T: Send + 'static>(
        &self,
        make: impl FnOnce(Result<T, String>) -> Done + Send + 'static,
    ) -> impl FnOnce(Result<T, steamworks::SteamError>) + Send + 'static {
        let done = self.done.clone();
        move |result| {
            if let Ok(mut done) = done.lock() {
                done.push(make(result.map_err(|error| error.to_string())));
            }
        }
    }
}

#[cfg(feature = "steam")]
impl Ugc for Steamworks {
    fn create(&mut self, request: u32) {
        let reply = self.reply(move |result: Result<(steamworks::PublishedFileId, bool), String>| Done::Created {
            request,
            result: result.map(|(item, legal)| (item.0, legal)),
        });
        self.ugc.create_item(self.app, steamworks::FileType::Community, reply);
    }

    fn submit(&mut self, request: u32, item: u64, update: &Update) {
        use steamworks::PublishedFileVisibility as Seen;
        let visibility = match update.visibility {
            Visibility::Public => Seen::Public,
            Visibility::Friends => Seen::FriendsOnly,
            Visibility::Private => Seen::Private,
            Visibility::Unlisted => Seen::Unlisted,
        };
        let mut handle = self
            .ugc
            .start_item_update(self.app, steamworks::PublishedFileId(item))
            .title(&update.title)
            .content_path(&update.content)
            .visibility(visibility)
            .tags(update.tags.clone(), false);
        if let Some(description) = &update.description {
            handle = handle.description(description);
        }
        if let Some(preview) = &update.preview {
            handle = handle.preview_path(preview);
        }
        let reply = self.reply(move |result: Result<(steamworks::PublishedFileId, bool), String>| Done::Submitted {
            request,
            result: result.map(|(item, legal)| (item.0, legal)),
        });
        // Steam reports progress through the handle; the answer is enough.
        let _ = handle.submit(update.note.as_deref(), reply);
    }

    fn subscribe(&mut self, request: u32, item: u64) {
        let reply = self.reply(move |result| Done::Subscribed { request, result });
        self.ugc.subscribe_item(steamworks::PublishedFileId(item), reply);
    }

    fn unsubscribe(&mut self, request: u32, item: u64) {
        let reply = self.reply(move |result| Done::Unsubscribed { request, result });
        self.ugc.unsubscribe_item(steamworks::PublishedFileId(item), reply);
    }

    fn subscribed(&mut self) -> Vec<u64> {
        self.ugc.subscribed_items(false).into_iter().map(|item| item.0).collect()
    }

    fn installed(&mut self, item: u64) -> Option<PathBuf> {
        let state = self.ugc.item_state(steamworks::PublishedFileId(item));
        if !state.contains(steamworks::ItemState::INSTALLED) {
            return None;
        }
        self.ugc.item_install_info(steamworks::PublishedFileId(item)).map(|info| PathBuf::from(info.folder))
    }

    fn download(&mut self, item: u64) -> bool {
        self.ugc.download_item(steamworks::PublishedFileId(item), true)
    }

    fn finished(&mut self) -> Vec<Done> {
        self.done.lock().map(|mut done| std::mem::take(&mut *done)).unwrap_or_default()
    }
}

/// Where the player's uploads are staged: beside their scene.
pub fn staging(scene: &Path) -> PathBuf {
    scene.parent().map_or_else(|| PathBuf::from("workshop-uploads"), |data| data.join("workshop-uploads"))
}

#[cfg(test)]
mod tests;
