//! What the game runs on: the operating system, the player's locale and the
//! device (a Steam Deck, or not), as text the game can show or act on.
use pen_protocol::{platform_op, service};

use super::{Answer, Request, Service};

pub struct Platform;

/// A POSIX locale (`en_US.UTF-8`, `sv_SE`) as a BCP 47 tag (`en-US`), or
/// None for none (`C`, `POSIX`).
#[cfg(any(not(windows), test))]
pub fn bcp47(posix: &str) -> Option<String> {
    let tag = posix.split(['.', '@']).next()?.replace('_', "-");
    let valid = !tag.is_empty() && tag != "C" && tag != "POSIX" && tag.chars().all(|char| char.is_ascii_alphanumeric() || char == '-');
    valid.then_some(tag)
}

/// The player's locale as a BCP 47 tag.
pub fn locale() -> Option<String> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::Globalization::GetUserDefaultLocaleName;
        let mut name = [0u16; 85];
        // SAFETY: the buffer is LOCALE_NAME_MAX_LENGTH (85) wide characters.
        let length = unsafe { GetUserDefaultLocaleName(name.as_mut_ptr(), name.len() as i32) };
        (length > 1).then(|| String::from_utf16_lossy(&name[..length as usize - 1]))
    }
    #[cfg(not(windows))]
    {
        ["LC_ALL", "LC_MESSAGES", "LANG"]
            .iter()
            .filter_map(|name| std::env::var(name).ok())
            .find(|value| !value.is_empty())
            .and_then(|value| bcp47(&value))
    }
}

/// Whether this is a Steam Deck: Steam says so in its launch environment,
/// or the board is Valve's Jupiter (LCD) or Galileo (OLED).
pub fn steam_deck() -> bool {
    if std::env::var("SteamDeck").is_ok_and(|value| value == "1") {
        return true;
    }
    let read = |name: &str| std::fs::read_to_string(format!("/sys/class/dmi/id/{name}")).unwrap_or_default();
    read("board_vendor").trim() == "Valve" && matches!(read("product_name").trim(), "Jupiter" | "Galileo")
}

impl Service for Platform {
    fn code(&self) -> u32 {
        service::PLATFORM
    }

    fn call(&mut self, request: &Request) -> Option<Answer> {
        Some(match request.operation {
            platform_op::OS => Answer::text(std::env::consts::OS),
            platform_op::LOCALE => match locale() {
                Some(tag) => Answer::text(&tag),
                None => Answer::unavailable(),
            },
            platform_op::DEVICE => {
                let deck = steam_deck();
                Answer { value: if deck { 1.0 } else { 0.0 }, ..Answer::text(if deck { "Steam Deck" } else { "desktop" }) }
            }
            _ => Answer::unknown(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posix_locales_become_tags() {
        assert_eq!(bcp47("en_US.UTF-8").as_deref(), Some("en-US"));
        assert_eq!(bcp47("sv_SE").as_deref(), Some("sv-SE"));
        assert_eq!(bcp47("de_DE@euro").as_deref(), Some("de-DE"));
        assert_eq!(bcp47("C.UTF-8"), None);
        assert_eq!(bcp47("POSIX"), None);
    }
}
