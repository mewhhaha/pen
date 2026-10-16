//! Steam as a service: achievements, stats and rich presence, through the
//! runtime's Steam client (`crate::steam`). Without one (a runtime built
//! without the `steam` feature, or Steam off) every request is answered
//! unavailable, and the game plays on.
use pen_protocol::service;

use super::{Answer, Request, Service};

pub struct SteamService {
    #[cfg(feature = "steam")]
    steam: Option<crate::steam::Steam>,
}

impl SteamService {
    /// No Steam: every request is answered unavailable.
    pub fn off() -> SteamService {
        SteamService {
            #[cfg(feature = "steam")]
            steam: None,
        }
    }

    /// The client `Steam::init` connected, if any.
    #[cfg(feature = "steam")]
    pub fn new(steam: Option<crate::steam::Steam>) -> SteamService {
        SteamService { steam }
    }
}

impl Service<crate::input::Input> for SteamService {
    fn code(&self) -> u32 {
        service::STEAM
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        #[cfg(feature = "steam")]
        if let Some(steam) = &mut self.steam {
            return Some(answer(steam, request));
        }
        let _ = request;
        Some(Answer::unavailable())
    }

    #[cfg(feature = "steam")]
    fn frame(&mut self, input: &mut crate::input::Input) {
        if let Some(steam) = &mut self.steam {
            steam.frame(input);
        }
    }
}

/// A whole number for an integer stat.
#[cfg(feature = "steam")]
#[allow(clippy::cast_possible_truncation, reason = "checked whole and within 2^24")]
fn whole(value: f32) -> anyhow::Result<i32> {
    anyhow::ensure!(
        value.fract() == 0.0 && value.abs() <= 16_777_216.0,
        "integer stats take whole numbers, not {value}"
    );
    Ok(value as i32)
}

#[cfg(feature = "steam")]
fn answer(steam: &mut crate::steam::Steam, request: &Request) -> Answer {
    use pen_protocol::steam_op;
    let name = request.text();
    let [a, ..] = request.numbers;
    let result = match request.operation {
        steam_op::UNLOCK_ACHIEVEMENT => steam.unlock(&name).map(|()| 1.0),
        steam_op::CLEAR_ACHIEVEMENT => steam.clear(&name).map(|()| 0.0),
        steam_op::SET_STAT => whole(a).and_then(|value| steam.set_stat(&name, value)).map(|()| a),
        steam_op::SET_STAT_FLOAT => steam.set_stat_float(&name, a).map(|()| a),
        steam_op::ADD_STAT => whole(a).and_then(|amount| steam.add_stat(&name, amount)).map(|value| value as f32),
        steam_op::GET_STAT => steam.stat(&name).map(|value| value as f32).or_else(|_| steam.stat_float(&name)),
        steam_op::SET_PRESENCE => {
            // The key is the text's first `a` code points, the value the rest.
            let split = pen_protocol::boundary::whole(a).map_or(0, |split| split as usize).min(request.values.len());
            let key = pen_runtime::services::text(&request.values[..split]);
            let value = pen_runtime::services::text(&request.values[split..]);
            steam.set_presence(&key, &value).map(|()| 0.0)
        }
        steam_op::CLEAR_PRESENCE => {
            steam.clear_presence();
            Ok(0.0)
        }
        steam_op::STATUS => Ok(1.0),
        _ => return Answer::unknown(),
    };
    match result {
        Ok(value) => Answer::ok(value),
        Err(error) => {
            eprintln!("Steam: {error:#}");
            Answer::failed()
        }
    }
}
