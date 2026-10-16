//! The one file watcher of a development session: the plugin builds of the
//! game and the dev tools (`plugin_build`), the game's build and assets
//! (`Dev`), the asset recipes' sources (`pipeline`) and the dev tools' build
//! (`devtools`) all subscribe to it. Each subscription names the
//! directories it reads, recursively or not, and is handed the events under
//! them, their paths spelled as it spelled the directories; a directory is
//! watched once however many subscriptions want it, and no longer once none
//! does.
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use anyhow::{Context, Result, anyhow};
use notify::{RecursiveMode, Watcher};

/// A change under a subscription's directories, or the watcher's trouble.
pub type Event = std::result::Result<notify::Event, String>;

/// Hands a subscription its events, on the watcher's thread: it should only
/// pass them on (over a channel).
type Deliver = Box<dyn Fn(Event) + Send>;

/// A directory and whether everything below it is watched too.
type Root = (PathBuf, bool);

struct Route {
    id: u64,
    /// Its directories as they resolve, each with how it was spelled.
    roots: Vec<(Root, PathBuf)>,
    deliver: Deliver,
}

impl Route {
    /// The event's paths as the route spelled its directories, if any of
    /// them is under one.
    fn spelled(&self, event: &notify::Event) -> Option<notify::Event> {
        let mut seen = false;
        let paths = event
            .paths
            .iter()
            .map(|path| match self.roots.iter().find(|(root, _)| under(path, root)) {
                Some(((resolved, _), given)) => {
                    seen = true;
                    path.strip_prefix(resolved).map_or_else(|_| path.clone(), |rest| given.join(rest))
                }
                None => path.clone(),
            })
            .collect();
        seen.then(|| notify::Event { paths, ..event.clone() })
    }
}

/// The roots as they resolve, with how they were spelled.
fn resolve(roots: &[(&Path, bool)]) -> Result<Vec<(Root, PathBuf)>> {
    roots
        .iter()
        .map(|(path, recursive)| {
            let resolved = std::fs::canonicalize(path).with_context(|| format!("watching {}", path.display()))?;
            Ok(((resolved, *recursive), path.to_path_buf()))
        })
        .collect()
}

struct Watching {
    watcher: notify::RecommendedWatcher,
    /// What the watcher watches now: `cover` of every route's roots.
    watched: Vec<Root>,
    next: u64,
}

struct Hub {
    /// Read by the watcher's thread for each event, so never held while the
    /// watcher is asked to watch (which waits on that thread).
    routes: Arc<Mutex<Vec<Route>>>,
    watching: Mutex<Watching>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Whether `path` is under `root` as it is watched.
fn under(path: &Path, (root, recursive): &Root) -> bool {
    if *recursive { path.starts_with(root) } else { path == root || path.parent() == Some(root) }
}

/// The fewest watches that see everything `wanted`: each directory once,
/// recursively if any wants it so, and none inside one watched recursively.
fn cover(wanted: &[Root]) -> Vec<Root> {
    let mut merged: Vec<Root> = Vec::new();
    for (path, recursive) in wanted {
        match merged.iter_mut().find(|(other, _)| other == path) {
            Some((_, other)) => *other |= *recursive,
            None => merged.push((path.clone(), *recursive)),
        }
    }
    let covered =
        |path: &Path| merged.iter().any(|(other, recursive)| *recursive && other != path && path.starts_with(other));
    let mut cover: Vec<Root> = merged.iter().filter(|(path, _)| !covered(path)).cloned().collect();
    cover.sort();
    cover
}

impl Hub {
    fn new() -> Result<Hub> {
        let routes: Arc<Mutex<Vec<Route>>> = Arc::default();
        let read = routes.clone();
        let watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
            let routes = lock(&read);
            match event {
                Ok(event) => {
                    for route in routes.iter() {
                        if let Some(event) = route.spelled(&event) {
                            (route.deliver)(Ok(event));
                        }
                    }
                }
                Err(error) => {
                    for route in routes.iter() {
                        (route.deliver)(Err(error.to_string()));
                    }
                }
            }
        })?;
        Ok(Hub { routes, watching: Mutex::new(Watching { watcher, watched: Vec::new(), next: 0 }) })
    }

    /// Watches what the routes want now, and nothing else.
    fn rewatch(&self, watching: &mut Watching) -> Result<()> {
        let wanted: Vec<Root> =
            lock(&self.routes).iter().flat_map(|route| route.roots.iter().map(|(root, _)| root.clone())).collect();
        let cover = cover(&wanted);
        // Unwatched first: a recursive watch going takes those under it along.
        for gone in watching.watched.iter().filter(|root| !cover.contains(root)) {
            let _ = watching.watcher.unwatch(&gone.0);
        }
        watching.watched.retain(|root| cover.contains(root));
        for (path, recursive) in cover {
            if watching.watched.iter().any(|(watched, _)| *watched == path) {
                continue;
            }
            let mode = if recursive { RecursiveMode::Recursive } else { RecursiveMode::NonRecursive };
            watching.watcher.watch(&path, mode).with_context(|| format!("watching {}", path.display()))?;
            watching.watched.push((path, recursive));
        }
        Ok(())
    }
}

fn hub() -> Result<&'static Hub> {
    static HUB: OnceLock<std::result::Result<Hub, String>> = OnceLock::new();
    HUB.get_or_init(|| Hub::new().map_err(|error| format!("{error:#}")))
        .as_ref()
        .map_err(|error| anyhow!("cannot watch files: {error}"))
}

/// Events under some directories, while it is kept.
pub struct Subscription {
    id: u64,
}

/// Hands `deliver` every event under `roots` (each a directory, and whether
/// below it too) from now until the subscription is dropped.
pub fn subscribe(roots: &[(&Path, bool)], deliver: impl Fn(Event) + Send + 'static) -> Result<Subscription> {
    let roots = resolve(roots)?;
    let hub = hub()?;
    let mut watching = lock(&hub.watching);
    let id = watching.next;
    watching.next += 1;
    lock(&hub.routes).push(Route { id, roots, deliver: Box::new(deliver) });
    if let Err(error) = hub.rewatch(&mut watching) {
        lock(&hub.routes).retain(|route| route.id != id);
        let _ = hub.rewatch(&mut watching);
        return Err(error);
    }
    Ok(Subscription { id })
}

impl Subscription {
    /// Watches `roots` too, from now on.
    #[cfg(any(feature = "compiler", test))]
    pub fn extend(&self, roots: &[(&Path, bool)]) -> Result<()> {
        let mut roots = resolve(roots)?;
        let hub = hub()?;
        let mut watching = lock(&hub.watching);
        if let Some(route) = lock(&hub.routes).iter_mut().find(|route| route.id == self.id) {
            roots.retain(|root| !route.roots.contains(root));
            route.roots.extend(roots);
        }
        hub.rewatch(&mut watching)
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        let Ok(hub) = hub() else { return };
        let mut watching = lock(&hub.watching);
        lock(&hub.routes).retain(|route| route.id != self.id);
        if let Err(error) = hub.rewatch(&mut watching) {
            eprintln!("watching files: {error:#}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::channel;
    use std::time::Duration;

    use super::*;

    fn root(path: &str, recursive: bool) -> Root {
        (PathBuf::from(path), recursive)
    }

    #[test]
    fn watches_each_directory_once_and_none_inside_a_recursive_one() {
        let wanted = [
            root("/game/.pen/runtime", false),
            root("/game/src/assets", true),
            root("/game/src/assets", false),
            root("/game/src/assets/fonts", false),
            root("/game/models", false),
            root("/game/models", true),
            root("/game/.pen/runtime", false),
            root("/game/src/assetsmore", false),
        ];
        assert_eq!(
            cover(&wanted),
            [
                root("/game/.pen/runtime", false),
                root("/game/models", true),
                root("/game/src/assets", true),
                root("/game/src/assetsmore", false),
            ]
        );
        assert!(under(Path::new("/a/b/c.wasm"), &root("/a/b", false)));
        assert!(!under(Path::new("/a/b/c/d.png"), &root("/a/b", false)));
        assert!(under(Path::new("/a/b/c/d.png"), &root("/a/b", true)));
        assert!(!under(Path::new("/a/bc/d.png"), &root("/a/b", true)));
    }

    #[test]
    fn subscriptions_share_the_watcher_and_get_their_own_events() {
        let directory = std::env::temp_dir().join(format!("pen-watch-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(directory.join("assets/deep")).unwrap();
        let (assets_sender, assets) = channel();
        let (top_sender, top) = channel();
        let deep = subscribe(&[(&directory.join("assets"), true)], move |event| {
            let _ = assets_sender.send(event);
        })
        .unwrap();
        let shallow = subscribe(&[(&directory, false)], move |event| {
            let _ = top_sender.send(event);
        })
        .unwrap();
        let wait = |events: &std::sync::mpsc::Receiver<Event>, path: &Path| loop {
            let event = events.recv_timeout(Duration::from_secs(5)).expect("no event").unwrap();
            if event.paths.iter().any(|seen| seen == path) {
                break;
            }
        };
        let rock = directory.join("assets/deep/rock.png");
        std::fs::write(&rock, "1").unwrap();
        wait(&assets, &rock);
        let game = directory.join("game.wasm");
        std::fs::write(&game, "1").unwrap();
        wait(&top, &game);
        // Nothing of the other's: the game is not under assets, and the
        // rock is below what the shallow one watches.
        assert!(assets.try_iter().flatten().all(|event| !event.paths.contains(&game)));
        assert!(top.try_iter().flatten().all(|event| !event.paths.contains(&rock)));
        // One going leaves the other watching.
        drop(shallow);
        std::fs::write(&rock, "2").unwrap();
        wait(&assets, &rock);
        // More, later: a directory beside it.
        std::fs::create_dir_all(directory.join("src")).unwrap();
        deep.extend(&[(&directory.join("src"), false)]).unwrap();
        let main = directory.join("src/main.blot");
        std::fs::write(&main, "1").unwrap();
        wait(&assets, &main);
        drop(deep);
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn paths_come_as_the_subscription_spelled_its_directories() {
        let resolved = PathBuf::from("/home/me/game");
        let route = Route {
            id: 0,
            roots: vec![((resolved.clone(), true), PathBuf::from("/link/game"))],
            deliver: Box::new(|_| {}),
        };
        let event =
            notify::Event::new(notify::EventKind::Any).add_path(resolved.join("src/a.blot")).add_path("/else/b".into());
        let spelled = route.spelled(&event).unwrap();
        assert_eq!(spelled.paths, [PathBuf::from("/link/game/src/a.blot"), PathBuf::from("/else/b")]);
        let elsewhere = notify::Event::new(notify::EventKind::Any).add_path("/else/b".into());
        assert!(route.spelled(&elsewhere).is_none());
    }
}
