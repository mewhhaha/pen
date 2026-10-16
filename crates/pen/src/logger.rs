//! Library crates (pen-render) report through the `log` crate; this prints
//! their records on stderr. Warnings and errors always show. `RUST_LOG`
//! raises the level, everywhere (`RUST_LOG=debug`) or for one crate
//! (`RUST_LOG=pen_render=debug`, several separated by commas), and
//! PEN_FRAME_TIMES shows pen-render's pipeline compile times beside its
//! frame lines.

use log::{Level, LevelFilter, Log, Metadata, Record};

struct Logger {
    everywhere: LevelFilter,
    targets: Vec<(String, LevelFilter)>,
}

impl Logger {
    fn from_env() -> Logger {
        let mut logger = Logger { everywhere: LevelFilter::Warn, targets: Vec::new() };
        if std::env::var_os("PEN_FRAME_TIMES").is_some() {
            logger.targets.push(("pen_render".into(), LevelFilter::Debug));
        }
        for part in std::env::var("RUST_LOG").unwrap_or_default().split(',') {
            match part.trim().split_once('=') {
                Some((target, level)) => {
                    if let Ok(level) = level.parse() {
                        logger.targets.push((target.replace('-', "_"), level));
                    }
                }
                None => {
                    if let Ok(level) = part.trim().parse() {
                        logger.everywhere = level;
                    }
                }
            }
        }
        logger
    }

    fn level(&self, target: &str) -> LevelFilter {
        let named = self.targets.iter().filter(|(name, _)| target == name || target.starts_with(&format!("{name}::")));
        named.map(|(_, level)| *level).max().unwrap_or(LevelFilter::Off).max(self.everywhere)
    }

    fn most(&self) -> LevelFilter {
        self.targets.iter().map(|(_, level)| *level).max().unwrap_or(LevelFilter::Off).max(self.everywhere)
    }
}

impl Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.level(metadata.target())
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let crate_name = record.target().split("::").next().unwrap_or_default();
        match record.level() {
            Level::Error | Level::Warn => {
                eprintln!("{crate_name}: {}: {}", record.level().as_str().to_lowercase(), record.args())
            }
            _ => eprintln!("{crate_name}: {}", record.args()),
        }
    }

    fn flush(&self) {}
}

/// Installs the logger; once, at startup.
pub fn install() {
    let logger = Logger::from_env();
    let most = logger.most();
    if log::set_boxed_logger(Box::new(logger)).is_ok() {
        log::set_max_level(most);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_crate_takes_its_own_level_or_the_general_one() {
        let logger = Logger { everywhere: LevelFilter::Warn, targets: vec![("pen_render".into(), LevelFilter::Debug)] };
        assert_eq!(logger.level("pen_render"), LevelFilter::Debug);
        assert_eq!(logger.level("pen_render::loader"), LevelFilter::Debug);
        assert_eq!(logger.level("pen_renderer"), LevelFilter::Warn);
        assert_eq!(logger.level("wgpu_core"), LevelFilter::Warn);
        assert_eq!(logger.most(), LevelFilter::Debug);
    }
}
