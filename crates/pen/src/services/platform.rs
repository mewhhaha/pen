//! What the game runs on here: the operating system, the player's locale and
//! the device (a Steam Deck, or not), which the platform service
//! (`pen_runtime::services::Platform`) answers with.
use pen_runtime::services::Platform;

/// A POSIX locale (`en_US.UTF-8`, `sv_SE`) as a BCP 47 tag (`en-US`), or
/// None for none (`C`, `POSIX`).
#[cfg(any(not(windows), test))]
pub fn bcp47(posix: &str) -> Option<String> {
    let tag = posix.split(['.', '@']).next()?.replace('_', "-");
    let valid = !tag.is_empty()
        && tag != "C"
        && tag != "POSIX"
        && tag.chars().all(|char| char.is_ascii_alphanumeric() || char == '-');
    valid.then_some(tag)
}

/// The player's locale as a BCP 47 tag.
pub fn locale() -> Option<String> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::Globalization::GetUserDefaultLocaleName;
        let mut name = [0u16; 85];
        // SAFETY: the buffer is LOCALE_NAME_MAX_LENGTH (85) wide characters.
        let length = unsafe { GetUserDefaultLocaleName(name.as_mut_ptr(), 85) };
        // The length counts the closing NUL; 0 is a failure.
        let length = usize::try_from(length).ok().filter(|&length| length > 1)?;
        Some(String::from_utf16_lossy(&name[..length - 1]))
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

/// The platform service for this machine.
pub fn platform() -> Platform {
    let deck = steam_deck();
    Platform {
        os: std::env::consts::OS.to_owned(),
        locale: locale(),
        device: if deck { "Steam Deck" } else { "desktop" }.to_owned(),
        handheld: deck,
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
