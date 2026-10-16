//! Steam, when the game runs under it (the `steam` feature). The overlay
//! needs nothing here: Steam injects it into the process and hooks the
//! swapchain the renderer presents to. The app ID comes from Steam's launch
//! environment, or `steam_appid.txt` beside the executable while developing
//! (480, Spacewar, for testing).
//!
//! Achievements and stats go by their API names on the app's Steamworks
//! page. Changing one changes it in Steam's copy of the player's stats, which
//! the Steam client loads before the game starts (SDK 1.61 dropped
//! `RequestCurrentStats`); [`Steam::frame`] stores changes with Steam (an
//! unlocked achievement at once, so its notification shows, stats at most
//! once a minute, and anything left when the game closes). The game asks for
//! them, and sets rich presence, through its services (`services::steam`,
//! gdev's `packages/services.blot`), which answer unavailable without Steam.
//! PEN_STEAM_DEBUG runs steps once Steam says the stats are in, or after two
//! seconds, for testing against Spacewar:
//!
//! ```text
//! PEN_STEAM_DEBUG="list unlock=ACH_WIN_ONE_GAME add=NumGames=1 show=NumGames" pen
//! ```
//!
//! Steps: `list` (every achievement and whether it is unlocked),
//! `unlock=NAME`, `clear=NAME`, `stat=NAME=N` (an integer stat),
//! `statf=NAME=X` (a float stat), `add=NAME=N` (to an integer stat),
//! `show=NAME` (a stat's value) and `reset` (every stat and achievement).
//! Without a running Steam client the game plays on without Steam.
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use pen_protocol::GameEvent;
use steamworks::{CallbackResult, Client};

use crate::input::Input;

/// How often changed stats are stored with Steam, which rate-limits it.
const STORE_INTERVAL: Duration = Duration::from_secs(60);
/// How long PEN_STEAM_DEBUG waits for Steam to say the stats are in.
const DEBUG_WAIT: Duration = Duration::from_secs(2);

/// Whether to start Steam outside a packaged game: `PEN_STEAM`, or
/// `PEN_STEAM_DEBUG` steps to run.
pub fn requested() -> bool {
    std::env::var_os("PEN_STEAM").is_some() || std::env::var_os("PEN_STEAM_DEBUG").is_some()
}

/// A `PEN_STEAM_DEBUG` step.
enum Step {
    List,
    Unlock(String),
    Clear(String),
    Stat(String, i32),
    StatFloat(String, f32),
    Add(String, i32),
    Show(String),
    Reset,
}

fn parse_steps(text: &str) -> Result<Vec<Step>> {
    let named = |value: Option<&str>, word: &str| -> Result<(String, String)> {
        let (name, number) =
            value.and_then(|value| value.split_once('=')).with_context(|| format!("{word} takes NAME=VALUE"))?;
        Ok((name.to_owned(), number.to_owned()))
    };
    text.split([' ', ',', '\n'])
        .filter(|word| !word.is_empty())
        .map(|word| {
            let (key, value) = match word.split_once('=') {
                Some((key, value)) => (key, Some(value)),
                None => (word, None),
            };
            let name = || value.map(str::to_owned).with_context(|| format!("{word} takes a NAME"));
            Ok(match key {
                "list" => Step::List,
                "reset" => Step::Reset,
                "unlock" => Step::Unlock(name()?),
                "clear" => Step::Clear(name()?),
                "show" => Step::Show(name()?),
                "stat" | "add" => {
                    let (name, number) = named(value, word)?;
                    let number = number.parse().with_context(|| format!("{word}: not an integer"))?;
                    if key == "stat" { Step::Stat(name, number) } else { Step::Add(name, number) }
                }
                "statf" => {
                    let (name, number) = named(value, word)?;
                    Step::StatFloat(name, number.parse().with_context(|| format!("{word}: not a number"))?)
                }
                _ => bail!("unknown PEN_STEAM_DEBUG step {word}"),
            })
        })
        .collect()
}

/// The error for a call Steam refused, about `what`.
fn refused(what: String) -> impl FnOnce(()) -> anyhow::Error {
    move |()| anyhow!("Steam has no {what} (or not the player's stats yet)")
}

pub struct Steam {
    client: Client,
    /// Whether Steam has said the player's stats are in, and since when it
    /// could have.
    received: bool,
    started: Instant,
    /// Changes not yet stored with Steam, and whether one should show now.
    unstored: bool,
    urgent: bool,
    stored: Instant,
    debug: Vec<Step>,
}

impl Steam {
    /// Connects to the running Steam client, or returns None (and plays on
    /// without Steam) when there is none.
    pub fn init() -> Option<Steam> {
        let debug = match std::env::var("PEN_STEAM_DEBUG").map(|text| parse_steps(&text)) {
            Ok(Ok(steps)) => steps,
            Ok(Err(error)) => {
                eprintln!("PEN_STEAM_DEBUG: {error:#}");
                Vec::new()
            }
            Err(_) => Vec::new(),
        };
        // A store build started outside Steam relaunches through it.
        if let Some(app_id) = std::env::var("PEN_STEAM_APP_ID").ok().and_then(|id| id.parse().ok())
            && steamworks::restart_app_if_necessary(steamworks::AppId(app_id))
        {
            std::process::exit(0);
        }
        let client = match Client::init() {
            Ok(client) => client,
            Err(error) => {
                eprintln!("Steam is off: {error}");
                if !debug.is_empty() {
                    eprintln!("PEN_STEAM_DEBUG: no achievements or stats without Steam");
                }
                return None;
            }
        };
        eprintln!("Steam: app {} for {}", client.utils().app_id().0, client.friends().name());
        if !debug.is_empty() {
            eprintln!("PEN_STEAM_DEBUG: waiting for the player's stats");
        }
        let now = Instant::now();
        Some(Steam { client, received: false, started: now, unstored: false, urgent: false, stored: now, debug })
    }

    /// Pumps Steam's callbacks once a frame and stores changes when due.
    /// Keys held when the overlay opened would never come up, so the game
    /// hears it as lost focus.
    pub fn frame(&mut self, input: &mut Input) {
        let mut results = Vec::new();
        self.client.process_callbacks(|result| results.push(result));
        let player = self.client.user().steam_id();
        for result in results {
            match result {
                CallbackResult::GameOverlayActivated(overlay) if overlay.active => {
                    input.enqueue(GameEvent::FocusLost);
                }
                CallbackResult::UserStatsReceived(received) if received.steam_id == player => match received.result {
                    Ok(()) => self.received = true,
                    Err(error) => eprintln!("Steam: the player's stats did not arrive: {error}"),
                },
                CallbackResult::UserStatsStored(stored) => {
                    if let Err(error) = stored.result {
                        eprintln!("Steam: storing stats failed: {error}");
                    }
                }
                CallbackResult::UserAchievementStored(achievement) => {
                    eprintln!("Steam: achievement {} stored", achievement.achievement_name);
                }
                _ => {}
            }
        }
        if !self.debug.is_empty() && (self.received || self.started.elapsed() >= DEBUG_WAIT) {
            self.run_debug();
        }
        if self.unstored && (self.urgent || self.stored.elapsed() >= STORE_INTERVAL) {
            self.store();
        }
    }

    fn changed(&mut self, urgent: bool) {
        self.unstored = true;
        self.urgent |= urgent;
    }

    /// Unlocks an achievement; Steam shows it once stored, which is soon.
    pub fn unlock(&mut self, name: &str) -> Result<()> {
        self.client.user_stats().achievement(name).set().map_err(refused(format!("achievement {name}")))?;
        self.changed(true);
        Ok(())
    }

    /// Locks an achievement again (for testing).
    pub fn clear(&mut self, name: &str) -> Result<()> {
        self.client.user_stats().achievement(name).clear().map_err(refused(format!("achievement {name}")))?;
        self.changed(true);
        Ok(())
    }

    /// Every achievement's API name and whether the player has it.
    pub fn achievements(&self) -> Result<Vec<(String, bool)>> {
        let stats = self.client.user_stats();
        let count = stats.get_num_achievements().map_err(|()| anyhow!("Steam lists no achievements"))?;
        let names = if count == 0 { Vec::new() } else { stats.get_achievement_names().unwrap_or_default() };
        Ok(names
            .into_iter()
            .map(|name| {
                let unlocked = stats.achievement(&name).get().unwrap_or(false);
                (name, unlocked)
            })
            .collect())
    }

    pub fn stat(&self, name: &str) -> Result<i32> {
        self.client.user_stats().get_stat_i32(name).map_err(refused(format!("integer stat {name}")))
    }

    pub fn stat_float(&self, name: &str) -> Result<f32> {
        self.client.user_stats().get_stat_f32(name).map_err(refused(format!("float stat {name}")))
    }

    /// Sets an integer stat (Steam refuses a decrease where the stat's
    /// page says it only increases).
    pub fn set_stat(&mut self, name: &str, value: i32) -> Result<()> {
        let stats = self.client.user_stats();
        stats.set_stat_i32(name, value).map_err(refused(format!("integer stat {name} that takes {value}")))?;
        self.changed(false);
        Ok(())
    }

    pub fn set_stat_float(&mut self, name: &str, value: f32) -> Result<()> {
        let stats = self.client.user_stats();
        stats.set_stat_f32(name, value).map_err(refused(format!("float stat {name} that takes {value}")))?;
        self.changed(false);
        Ok(())
    }

    /// Adds to an integer stat; returns its new value.
    pub fn add_stat(&mut self, name: &str, amount: i32) -> Result<i32> {
        let value = self.stat(name)?.saturating_add(amount);
        self.set_stat(name, value)?;
        Ok(value)
    }

    /// Sets a rich presence key, which friends see (`status` in the game's
    /// details, `steam_display` through the app's localization tokens).
    pub fn set_presence(&mut self, key: &str, value: &str) -> Result<()> {
        if !self.client.friends().set_rich_presence(key, Some(value)) {
            bail!("Steam refused rich presence {key}={value}");
        }
        Ok(())
    }

    /// Clears every rich presence key.
    pub fn clear_presence(&mut self) {
        self.client.friends().clear_rich_presence();
    }

    /// Resets every stat and, with `achievements`, every achievement (for testing).
    pub fn reset(&mut self, achievements: bool) -> Result<()> {
        self.client.user_stats().reset_all_stats(achievements).map_err(|()| anyhow!("Steam refused the reset"))?;
        self.changed(true);
        Ok(())
    }

    /// Stores changed stats and achievements with Steam now.
    pub fn store(&mut self) {
        if self.client.user_stats().store_stats().is_err() {
            eprintln!("Steam: storing stats failed");
        }
        self.unstored = false;
        self.urgent = false;
        self.stored = Instant::now();
    }

    fn run_debug(&mut self) {
        for step in std::mem::take(&mut self.debug) {
            let result = match &step {
                Step::List => self.achievements().map(|achievements| {
                    for (name, unlocked) in achievements {
                        eprintln!("PEN_STEAM_DEBUG: {name}: {}", if unlocked { "unlocked" } else { "locked" });
                    }
                }),
                Step::Unlock(name) => self.unlock(name).map(|()| eprintln!("PEN_STEAM_DEBUG: unlocked {name}")),
                Step::Clear(name) => self.clear(name).map(|()| eprintln!("PEN_STEAM_DEBUG: cleared {name}")),
                Step::Stat(name, value) => {
                    self.set_stat(name, *value).map(|()| eprintln!("PEN_STEAM_DEBUG: {name} = {value}"))
                }
                Step::StatFloat(name, value) => {
                    self.set_stat_float(name, *value).map(|()| eprintln!("PEN_STEAM_DEBUG: {name} = {value}"))
                }
                Step::Add(name, amount) => {
                    self.add_stat(name, *amount).map(|value| eprintln!("PEN_STEAM_DEBUG: {name} = {value}"))
                }
                Step::Show(name) => {
                    let value = self.stat(name).map(|value| value.to_string());
                    value.or_else(|_| self.stat_float(name).map(|value| value.to_string())).map(|value| {
                        eprintln!("PEN_STEAM_DEBUG: {name} = {value}");
                    })
                }
                Step::Reset => {
                    self.reset(true).map(|()| eprintln!("PEN_STEAM_DEBUG: reset every stat and achievement"))
                }
            };
            if let Err(error) = result {
                eprintln!("PEN_STEAM_DEBUG: {error:#}");
            }
        }
    }
}

impl Drop for Steam {
    /// Stores what is left before Steam shuts down with the client.
    fn drop(&mut self) {
        if self.unstored {
            self.store();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_steps() {
        let steps = parse_steps("list unlock=ACH_WIN_ONE_GAME,add=NumGames=2 statf=FeetTraveled=1.5 reset").unwrap();
        assert_eq!(steps.len(), 5);
        assert!(matches!(&steps[2], Step::Add(name, 2) if name == "NumGames"));
        assert!(matches!(&steps[3], Step::StatFloat(name, value) if name == "FeetTraveled" && *value == 1.5));
        assert!(parse_steps("unlock").is_err());
        assert!(parse_steps("add=NumGames").is_err());
        assert!(parse_steps("launch").is_err());
    }
}
