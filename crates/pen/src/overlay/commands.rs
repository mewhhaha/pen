//! The console's commands. The game's are read from the checkout when the
//! overlay starts, so they stay in step with it: `commands.ts` names each
//! (`export const <name>Command = N;`, with its doc comment) and the values
//! some take (`<name>Kinds`, as `rockKinds` lists the rocks), and the browser
//! console in `web/client.ts`, when it is there, gives each its short
//! description and its argument's range and default, and the controls
//! `/help` lists. The parser is tolerant: what it does not recognise is left
//! out, and `/command N [ARG]` sends any command by number.
//!
//! The console's own commands are the browser's (`/help`, `/clear`,
//! `/diagnostics`, `/tune`) and the native runtime's time travel, input
//! recording and overlay screenshots.
use std::path::Path;

/// One of the values a command's argument takes; it is sent as its index.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub value: String,
    pub about: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Argument {
    /// None; the game gets 0.
    None,
    /// One of these, sent as its index (`rockKinds`).
    Choice(Vec<Choice>),
    /// A whole number in `min..=max`, `default` when left out (the
    /// browser's `usage: /gnorps [1-50]` and `Number(args[0] ?? "1")`).
    Count { min: i64, max: i64, default: i64 },
    /// Unknown: any number, 0 when left out.
    Number,
}

/// A command the game takes as `GameEvent::Console`.
#[derive(Clone, Debug, PartialEq)]
pub struct Command {
    pub name: String,
    pub code: u32,
    /// What it does, in a line.
    pub about: String,
    pub argument: Argument,
}

/// The game's commands and controls, as read from a checkout.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Table {
    pub commands: Vec<Command>,
    /// The controls `/help` lists, as (keys, action); an empty action
    /// starts a group, and an empty pair is a gap.
    pub controls: Vec<(String, String)>,
}

impl Table {
    /// The checkout's table; empty when it has no `commands.ts`.
    pub fn load(root: &Path) -> Table {
        let Ok(commands) = std::fs::read_to_string(root.join("commands.ts")) else { return Table::default() };
        let client = std::fs::read_to_string(root.join("web/client.ts")).ok();
        Table::parse(&commands, client.as_deref())
    }

    pub fn parse(commands: &str, client: Option<&str>) -> Table {
        let kinds = declarations(commands, "Kinds")
            .filter_map(|(name, _, value)| Some((name, choices(value)?)))
            .collect::<Vec<_>>();
        let commands = declarations(commands, "Command")
            .filter_map(|(name, doc, value)| {
                let code = value.split([';', '\n']).next()?.trim().parse().ok()?;
                let details = client.and_then(|client| browser_command(client, &name)).unwrap_or_default();
                let argument = match kinds.iter().find(|(kind, _)| *kind == name) {
                    Some((_, choices)) => Argument::Choice(choices.clone()),
                    None => details.argument(),
                };
                let about = details.about.unwrap_or_else(|| first_sentence(&doc));
                Some(Command { name, code, about, argument })
            })
            .collect();
        let controls = client.and_then(controls).unwrap_or_default();
        Table { commands, controls }
    }

    pub fn command(&self, name: &str) -> Option<&Command> {
        self.commands.iter().find(|command| command.name == name)
    }
}

/// Each `export const <name><suffix>` in `source`: the name, its doc
/// comment and the text after its `=`.
fn declarations<'a>(source: &'a str, suffix: &'a str) -> impl Iterator<Item = (String, String, &'a str)> + 'a {
    source.match_indices("export const ").filter_map(move |(at, keyword)| {
        let rest = &source[at + keyword.len()..];
        let length = rest.find(|char: char| !(char.is_ascii_alphanumeric() || char == '_'))?;
        let name = rest[..length].strip_suffix(suffix).filter(|name| !name.is_empty())?;
        let value = &rest[length..];
        let value = &value[value.find('=')? + 1..];
        Some((name.to_owned(), doc_before(&source[..at]), value.trim_start()))
    })
}

/// The `/** ... */` comment that ends `before`, as one line of text.
fn doc_before(before: &str) -> String {
    let before = before.trim_end();
    let Some(body) = before.strip_suffix("*/") else { return String::new() };
    let Some(start) = body.rfind("/**") else { return String::new() };
    body[start + 3..]
        .lines()
        .map(|line| line.trim().trim_start_matches('*').trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn first_sentence(doc: &str) -> String {
    match doc.find(". ") {
        Some(end) => doc[..=end].to_owned(),
        None => doc.to_owned(),
    }
}

/// The string literal `text` starts with (after whitespace), and the text
/// after it.
fn string_literal(text: &str) -> Option<(String, &str)> {
    let text = text.trim_start();
    let quote = text.chars().next().filter(|char| matches!(char, '"' | '\'' | '`'))?;
    let mut value = String::new();
    let mut chars = text.char_indices().skip(1);
    while let Some((index, char)) = chars.next() {
        match char {
            '\\' => {
                if let Some((_, escaped)) = chars.next() {
                    value.push(if escaped == 'n' { '\n' } else { escaped });
                }
            }
            char if char == quote => return Some((value, &text[index + 1..])),
            char => value.push(char),
        }
    }
    None
}

/// What lies between the bracket `text` starts with and the one that
/// closes it, skipping brackets in strings and comments.
fn bracketed(text: &str, open: char, close: char) -> Option<&str> {
    let text = text.trim_start();
    if !text.starts_with(open) {
        return None;
    }
    let mut depth = 0;
    let mut rest = text;
    while let Some(char) = rest.chars().next() {
        let at = text.len() - rest.len();
        if matches!(char, '"' | '\'' | '`') {
            rest = string_literal(rest)?.1;
            continue;
        }
        if rest.starts_with("//") {
            rest = rest.find('\n').map_or("", |end| &rest[end..]);
            continue;
        }
        if rest.starts_with("/*") {
            rest = rest.find("*/").map_or("", |end| &rest[end + 2..]);
            continue;
        }
        if char == open {
            depth += 1;
        } else if char == close {
            depth -= 1;
            if depth == 0 {
                return Some(&text[1..at]);
            }
        }
        rest = &rest[char.len_utf8()..];
    }
    None
}

/// Every string literal in `text`, in order.
fn strings(mut text: &str) -> Vec<String> {
    let mut found = Vec::new();
    while let Some(at) = text.find(['"', '\'', '`']) {
        match string_literal(&text[at..]) {
            Some((value, rest)) => {
                found.push(value);
                text = rest;
            }
            None => break,
        }
    }
    found
}

/// `[{ value: "...", about: "..." }, ...]` as choices.
fn choices(value: &str) -> Option<Vec<Choice>> {
    let mut list = bracketed(value, '[', ']')?;
    let mut choices = Vec::new();
    while let Some(start) = list.find('{') {
        let object = bracketed(&list[start..], '{', '}')?;
        let field = |name: &str| {
            let at = object.find(&format!("{name}:"))?;
            string_literal(&object[at + name.len() + 1..]).map(|(value, _)| value)
        };
        if let Some(value) = field("value") {
            choices.push(Choice { value, about: field("about").unwrap_or_default() });
        }
        list = &list[start + object.len() + 2..];
    }
    (!choices.is_empty()).then_some(choices)
}

/// What the browser console says about a command.
#[derive(Default)]
struct BrowserCommand {
    found: bool,
    about: Option<String>,
    /// What follows `usage: /<name>`.
    usage: Option<String>,
    /// `Number(args[0] ?? "N")`.
    default: Option<i64>,
}

impl BrowserCommand {
    fn argument(&self) -> Argument {
        let Some(usage) = &self.usage else {
            // A command whose run never checks its arguments takes none.
            return if self.found { Argument::None } else { Argument::Number };
        };
        let usage = usage.trim();
        if usage.is_empty() {
            return Argument::None;
        }
        let range = usage.trim_start_matches('[').trim_end_matches(']').split_once('-');
        match range.and_then(|(min, max)| Some((min.trim().parse().ok()?, max.trim().parse().ok()?))) {
            Some((min, max)) => Argument::Count { min, max, default: self.default.unwrap_or(min).clamp(min, max) },
            None => Argument::Number,
        }
    }
}

/// The entry `<name>: { ... }` of client.ts's console commands.
fn browser_command(client: &str, name: &str) -> Option<BrowserCommand> {
    let console = &client[client.find("function createConsole")?..];
    let key = format!("{name}: {{");
    let at = console.match_indices(&key).map(|(at, _)| at).find(|at| {
        let line = &console[console[..*at].rfind('\n').map_or(0, |start| start + 1)..*at];
        line.trim().is_empty()
    })?;
    let block = bracketed(&console[at + key.len() - 1..], '{', '}')?;
    let about = block.find("about:").and_then(|at| string_literal(&block[at + 6..])).map(|(about, _)| about);
    let marker = format!("usage: /{name}");
    let usage = block.find(&marker).map(|at| {
        let rest = &block[at + marker.len()..];
        rest[..rest.find(['"', '\'', '`']).unwrap_or(rest.len())].to_owned()
    });
    let default =
        block.find("?? ").and_then(|at| string_literal(&block[at + 3..])).and_then(|(value, _)| value.parse().ok());
    Some(BrowserCommand { found: true, about, usage, default })
}

/// client.ts's `controls`, as (keys, action) pairs.
fn controls(client: &str) -> Option<Vec<(String, String)>> {
    let declaration = &client[client.find("const controls")?..];
    let list = bracketed(&declaration[declaration.find('=')? + 1..], '[', ']')?;
    let strings = strings(list);
    if strings.is_empty() || !strings.len().is_multiple_of(2) {
        return None;
    }
    Some(strings.chunks(2).map(|pair| (pair[0].clone(), pair[1].clone())).collect())
}

/// The profiler panel's levels, as the browser's `/diagnostics` offers them.
pub const LEVELS: [(&str, &str); 4] = [
    ("off", "hide the profiler"),
    ("low", "frame rate"),
    ("medium", "frame times, CPU work and a frame-time graph"),
    ("high", "everything: CPU phases, draws, memory, GPU"),
];

/// What a console line asks for.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// `GameEvent::Console`, and what to print once it is sent.
    Game {
        code: u32,
        argument: f32,
        echo: String,
    },
    Help,
    Clear,
    /// The profiler's level (an index into `LEVELS`); none asks for it.
    Diagnostics(Option<usize>),
    Pause,
    Play,
    Step(u32),
    /// Back this many seconds of game time.
    Rewind(f64),
    /// The last seconds again, then on live.
    Replay(f64),
    /// Starts recording input (to this file, or a new one) or stops.
    Record(Option<String>),
    /// A screenshot with the overlay (to this file, or a new one).
    Shot(Option<String>),
    /// A tunable's new value, by its id (the Tunables panel).
    Tune(u32, f32),
    /// `/tune [NAME [VALUE]]`: lists the tunables, shows one or sets it.
    Tunable(Option<String>, Option<f32>),
    /// Writes the game's tuned values into the checkout as the defaults.
    Bake,
}

/// The console's own commands: name, argument and description.
pub const BUILT_IN: [(&str, &str, &str); 12] = [
    ("help", "", "list the controls and these commands"),
    ("clear", "", "clear this console"),
    ("diagnostics", "off|low|medium|high", "the profiler panel (off hides it)"),
    ("command", "N [ARG]", "send the game command numbered N"),
    ("pause", "", "pause the game (Ctrl+Shift+P)"),
    ("play", "", "play on, live from here (Ctrl+Shift+P)"),
    ("step", "[N]", "run N frames and stay paused (Ctrl+Shift+period)"),
    ("rewind", "[SECONDS]", "go back that much game time, paused (1 s)"),
    ("replay", "[SECONDS]", "play the last seconds again, then live (5 s; Ctrl+Shift+R)"),
    ("record", "[FILE]", "start or stop recording input to a .penrec"),
    ("shot", "[FILE]", "save a screenshot with this overlay"),
    ("tune", "[NAME [VALUE]]", "list the live-tunable constants, or show or set one (the Tunables panel)"),
];

/// A command and what completes its argument.
pub struct Entry {
    pub name: String,
    pub about: String,
    pub options: Vec<Choice>,
}

/// What the console completes: its own commands and the game's, the game's
/// first when a name is both (except help, clear, diagnostics and command).
pub fn catalog(table: &Table) -> Vec<Entry> {
    let mut entries: Vec<Entry> = Vec::new();
    for (name, _, about) in BUILT_IN.iter().take(4) {
        entries.push(Entry { name: (*name).to_owned(), about: (*about).to_owned(), options: Vec::new() });
    }
    entries[2].options =
        LEVELS.iter().map(|(value, about)| Choice { value: (*value).to_owned(), about: (*about).to_owned() }).collect();
    for command in &table.commands {
        if entries.iter().all(|entry| entry.name != command.name) {
            let options = match &command.argument {
                Argument::Choice(choices) => choices.clone(),
                _ => Vec::new(),
            };
            entries.push(Entry { name: command.name.clone(), about: command.about.clone(), options });
        }
    }
    for (name, _, about) in BUILT_IN.iter().skip(4) {
        if entries.iter().all(|entry| entry.name != *name) {
            entries.push(Entry { name: (*name).to_owned(), about: (*about).to_owned(), options: Vec::new() });
        }
    }
    entries
}

/// A completion for the console's input (gdev's `web/completion.ts`).
#[derive(Clone, Debug, PartialEq)]
pub struct Suggestion {
    /// The input after choosing it.
    pub text: String,
    /// What the list shows, and how many of its first bytes were typed.
    pub label: String,
    pub matched: usize,
    pub about: String,
    /// Choosing it leaves a command ready to run, rather than one that
    /// still takes an argument.
    pub complete: bool,
}

/// What `input` could become: commands whose names start with what follows
/// the slash, or, after a command and a space, the values its argument takes
/// that start with what follows. Anything else has none.
pub fn suggest(input: &str, catalog: &[Entry]) -> Vec<Suggestion> {
    let Some(typed) = input.strip_prefix('/') else { return Vec::new() };
    let Some((name, typed)) = typed.split_once(' ') else {
        let mut entries: Vec<&Entry> = catalog.iter().filter(|entry| entry.name.starts_with(typed)).collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        return entries
            .into_iter()
            .map(|entry| Suggestion {
                text: format!("/{}{}", entry.name, if entry.options.is_empty() { "" } else { " " }),
                label: format!("/{}", entry.name),
                matched: typed.len() + 1,
                about: entry.about.clone(),
                complete: entry.options.is_empty(),
            })
            .collect();
    };
    let Some(entry) = catalog.iter().find(|entry| entry.name == name) else { return Vec::new() };
    if typed.contains(' ') {
        return Vec::new();
    }
    entry
        .options
        .iter()
        .filter(|choice| choice.value.starts_with(typed))
        .map(|choice| Suggestion {
            text: format!("/{name} {}", choice.value),
            label: choice.value.clone(),
            matched: typed.len(),
            about: choice.about.clone(),
            complete: true,
        })
        .collect()
}

fn number(text: &str, usage: &str) -> Result<f64, String> {
    text.parse::<f64>().ok().filter(|value| value.is_finite()).ok_or_else(|| usage.to_owned())
}

/// What the console line `text` asks for, or why it cannot be done.
pub fn parse(text: &str, table: &Table) -> Result<Action, String> {
    let unknown = || format!("unknown command {text}; try /help");
    let words: Vec<&str> = text.strip_prefix('/').ok_or_else(unknown)?.split_whitespace().collect();
    let (&name, args) = words.split_first().ok_or_else(unknown)?;
    match (name, args) {
        ("help", _) => return Ok(Action::Help),
        ("clear", _) => return Ok(Action::Clear),
        ("diagnostics", []) => return Ok(Action::Diagnostics(None)),
        ("diagnostics", [level]) => {
            if let Some(index) = LEVELS.iter().position(|(name, _)| name == level) {
                return Ok(Action::Diagnostics(Some(index)));
            }
            return Err("usage: /diagnostics off|low|medium|high".into());
        }
        ("diagnostics", _) => return Err("usage: /diagnostics off|low|medium|high".into()),
        ("command", [code, rest @ ..]) if rest.len() <= 1 => {
            let usage = "usage: /command N [ARG]: N a command's number, ARG a number";
            let code = code.parse::<u32>().map_err(|_| usage.to_owned())?;
            let argument = rest.first().map(|arg| number(arg, usage)).transpose()?.unwrap_or(0.0) as f32;
            return Ok(Action::Game { code, argument, echo: format!("command {code}: {argument}") });
        }
        ("command", _) => return Err("usage: /command N [ARG]".into()),
        ("tune", []) => return Ok(Action::Tunable(None, None)),
        ("tune", [name]) => return Ok(Action::Tunable(Some((*name).to_owned()), None)),
        ("tune", [name, value]) => {
            let value = number(value, "usage: /tune [NAME [VALUE]]: VALUE a number")? as f32;
            return Ok(Action::Tunable(Some((*name).to_owned()), Some(value)));
        }
        ("tune", _) => return Err("usage: /tune [NAME [VALUE]]".into()),
        _ => {}
    }
    if let Some(command) = table.command(name) {
        return game(command, args);
    }
    let seconds = |args: &[&str], default: f64, usage: &str| -> Result<f64, String> {
        match args {
            [] => Ok(default),
            [value] => {
                number(value, usage).and_then(|value| if value > 0.0 { Ok(value) } else { Err(usage.to_owned()) })
            }
            _ => Err(usage.to_owned()),
        }
    };
    match (name, args) {
        ("pause", []) => Ok(Action::Pause),
        ("play", []) => Ok(Action::Play),
        ("step", []) => Ok(Action::Step(1)),
        ("step", [count]) => count
            .parse::<u32>()
            .ok()
            .filter(|count| (1..=600).contains(count))
            .map(Action::Step)
            .ok_or_else(|| "usage: /step [1-600]".into()),
        ("rewind", args) => seconds(args, 1.0, "usage: /rewind [SECONDS]").map(Action::Rewind),
        ("replay", args) => seconds(args, 5.0, "usage: /replay [SECONDS]").map(Action::Replay),
        ("record", []) => Ok(Action::Record(None)),
        ("record", [file]) => Ok(Action::Record(Some((*file).to_owned()))),
        ("shot", []) => Ok(Action::Shot(None)),
        ("shot", [file]) => Ok(Action::Shot(Some((*file).to_owned()))),
        ("pause" | "play" | "record" | "shot" | "step", _) => {
            let (_, argument, _) = BUILT_IN.iter().find(|(built_in, _, _)| *built_in == name).expect("a built-in");
            Err(format!("usage: /{name} {argument}").trim_end().to_owned())
        }
        _ => Err(unknown()),
    }
}

/// A game command with its arguments, checked as the browser checks them.
fn game(command: &Command, args: &[&str]) -> Result<Action, String> {
    let name = &command.name;
    let (argument, echo) = match &command.argument {
        Argument::None => {
            if !args.is_empty() {
                return Err(format!("usage: /{name}"));
            }
            (0.0, name.clone())
        }
        Argument::Choice(choices) => {
            let index = match args {
                [value] => choices.iter().position(|choice| choice.value == *value),
                _ => None,
            };
            let Some(index) = index else {
                let values: Vec<&str> = choices.iter().map(|choice| choice.value.as_str()).collect();
                return Err(format!("usage: /{name} {}", values.join("|")));
            };
            (index as f32, format!("{name}: {}", choices[index].value))
        }
        Argument::Count { min, max, default } => {
            let count = match args {
                [] => Some(*default),
                [value] => value.parse::<i64>().ok().filter(|count| (*min..=*max).contains(count)),
                _ => None,
            };
            let count = count.ok_or_else(|| format!("usage: /{name} [{min}-{max}]"))?;
            (count as f32, format!("{name}: {count}"))
        }
        Argument::Number => {
            let usage = format!("usage: /{name} [NUMBER]");
            let value = match args {
                [] => 0.0,
                [value] => number(value, &usage)?,
                _ => return Err(usage),
            };
            (value as f32, format!("{name}: {value}"))
        }
    };
    Ok(Action::Game { code: command.code, argument, echo })
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMANDS: &str = r#"// The game's console commands.

/** Picks the rock (src/rock.blot) by its kind's index in rockKinds, the
 * order rock.kinds lists them; each kind's mesh is rock_<name>.mesh.json. */
export const rockCommand = 1;
/** Opens the menu's settings (src/menu.blot). */
export const settingsCommand = 2;
/** While playing, hires that many gnorps at the first stash for nothing
 * (src/hand.blot): a shortcut for trying things out. */
export const gnorpsCommand = 3;
/** While playing, puts that much salt in the first stash. */
export const saltCommand = 4;
export const windCommand = 6;
export const futureCommand = 9;

export const rockKinds: readonly { value: string; about: string }[] = [
  { value: "boulder", about: "an angular glacial erratic" },
  { value: "moai", about: "an Easter Island head" },
];
"#;

    const CLIENT: &str = r#"
const controls: readonly (readonly [string, string])[] = [
  ["Tab", "switch between editing and playing"],
  [
    "click a gnorp",
    "pick it up",
  ],
  ["", ""],
];

function createConsole() {
  const commands = {
    help: {
      about: "list the controls and these commands",
      options: [],
      run() {},
    },
    rock: {
      about: "change the gnorp rock",
      options: rockKinds,
      run(args) {
        throw new Error(`usage: /rock ${names.join("|")}`);
      },
    },
    settings: {
      about: "the game's settings (F2)",
      options: [],
      run() {
        toGame({ tag: "Console", value: { command: settingsCommand, argument: 0 } });
      },
    },
    gnorps: {
      about: "hire gnorps at the station for nothing, while playing",
      options: [],
      run(args) {
        const count = Number(args[0] ?? "1");
        if (args.length > 1) {
          throw new Error("usage: /gnorps [1-50]");
        }
      },
    },
    salt: {
      about: "put salt in the stash for nothing, while playing",
      run(args) {
        const amount = Number(args[0] ?? "100");
        throw new Error("usage: /salt [1-5000]");
      },
    },
    wind: {
      about: "send a gust of wind across at once, while playing",
      run(args) {
        if (args.length > 0) throw new Error("usage: /wind");
      },
    },
  };
}
"#;

    fn table() -> Table {
        Table::parse(COMMANDS, Some(CLIENT))
    }

    #[test]
    fn reads_commands_kinds_and_the_browser_console() {
        let table = table();
        let names: Vec<(&str, u32)> = table.commands.iter().map(|c| (c.name.as_str(), c.code)).collect();
        assert_eq!(names, [("rock", 1), ("settings", 2), ("gnorps", 3), ("salt", 4), ("wind", 6), ("future", 9)]);
        let rock = table.command("rock").unwrap();
        assert_eq!(rock.about, "change the gnorp rock");
        let Argument::Choice(kinds) = &rock.argument else { panic!("{:?}", rock.argument) };
        assert_eq!(kinds[1], Choice { value: "moai".into(), about: "an Easter Island head".into() });
        assert_eq!(table.command("settings").unwrap().argument, Argument::None);
        assert_eq!(table.command("gnorps").unwrap().argument, Argument::Count { min: 1, max: 50, default: 1 });
        assert_eq!(table.command("salt").unwrap().argument, Argument::Count { min: 1, max: 5000, default: 100 });
        assert_eq!(table.command("wind").unwrap().argument, Argument::None);
        // Not in the browser console: its doc comment, and any number.
        let future = table.command("future").unwrap();
        assert_eq!((future.about.as_str(), &future.argument), ("", &Argument::Number));
        assert_eq!(
            table.controls,
            [
                ("Tab".to_owned(), "switch between editing and playing".to_owned()),
                ("click a gnorp".to_owned(), "pick it up".to_owned()),
                (String::new(), String::new()),
            ]
        );
    }

    #[test]
    fn without_the_browser_console_doc_comments_describe() {
        let table = Table::parse(COMMANDS, None);
        let gnorps = table.command("gnorps").unwrap();
        assert_eq!(
            gnorps.about,
            "While playing, hires that many gnorps at the first stash for nothing (src/hand.blot): a shortcut for trying things out."
        );
        assert_eq!(gnorps.argument, Argument::Number);
        assert!(matches!(table.command("rock").unwrap().argument, Argument::Choice(_)));
        assert!(table.controls.is_empty());
        assert_eq!(Table::parse("garbage { [ \"", Some("also ( garbage")), Table::default());
    }

    #[test]
    fn checks_arguments_as_the_browser_does() {
        let table = table();
        let game = |code, argument: f32, echo: &str| Ok(Action::Game { code, argument, echo: echo.into() });
        assert_eq!(parse("/rock moai", &table), game(1, 1.0, "rock: moai"));
        assert_eq!(parse("/rock", &table), Err("usage: /rock boulder|moai".into()));
        assert_eq!(parse("/gnorps", &table), game(3, 1.0, "gnorps: 1"));
        assert_eq!(parse("/salt", &table), game(4, 100.0, "salt: 100"));
        assert_eq!(parse("/salt 6000", &table), Err("usage: /salt [1-5000]".into()));
        assert_eq!(parse("/wind", &table), game(6, 0.0, "wind"));
        assert_eq!(parse("/wind 2", &table), Err("usage: /wind".into()));
        assert_eq!(parse("/command 7 2.5", &table), game(7, 2.5, "command 7: 2.5"));
        assert_eq!(
            parse("/command x", &table),
            Err("usage: /command N [ARG]: N a command's number, ARG a number".into())
        );
        assert_eq!(parse("/diagnostics high", &table), Ok(Action::Diagnostics(Some(3))));
        assert_eq!(parse("/rewind", &table), Ok(Action::Rewind(1.0)));
        assert_eq!(parse("/replay 2", &table), Ok(Action::Replay(2.0)));
        assert_eq!(parse("/step 3", &table), Ok(Action::Step(3)));
        assert_eq!(parse("/pause now", &table), Err("usage: /pause".into()));
        assert_eq!(parse("/record a.penrec", &table), Ok(Action::Record(Some("a.penrec".into()))));
        assert_eq!(parse("/tune", &table), Ok(Action::Tunable(None, None)));
        assert_eq!(parse("/tune camera.zoom_speed", &table), Ok(Action::Tunable(Some("camera.zoom_speed".into()), None)));
        assert_eq!(parse("/tune wind.least_wait 2.5", &table), Ok(Action::Tunable(Some("wind.least_wait".into()), Some(2.5))));
        assert_eq!(parse("/tune a b", &table), Err("usage: /tune [NAME [VALUE]]: VALUE a number".into()));
        assert_eq!(parse("/tune a 1 2", &table), Err("usage: /tune [NAME [VALUE]]".into()));
        assert_eq!(parse("/nope", &table), Err("unknown command /nope; try /help".into()));
        assert_eq!(parse("hello", &table), Err("unknown command hello; try /help".into()));
    }

    #[test]
    fn suggests_commands_then_their_values() {
        let catalog = catalog(&table());
        let labels = |input| suggest(input, &catalog).into_iter().map(|s| s.label).collect::<Vec<_>>();
        assert_eq!(labels("/s"), ["/salt", "/settings", "/shot", "/step"]);
        assert_eq!(labels("/ro"), ["/rock"]);
        assert_eq!(labels("/rock "), ["boulder", "moai"]);
        assert_eq!(labels("/rock m"), ["moai"]);
        assert_eq!(labels("/diagnostics h"), ["high"]);
        assert!(labels("rock").is_empty());
        assert!(labels("/rock moai x").is_empty());
        let rock = &suggest("/ro", &catalog)[0];
        assert_eq!((rock.text.as_str(), rock.matched, rock.complete), ("/rock ", 3, false));
        let moai = &suggest("/rock m", &catalog)[0];
        assert_eq!((moai.text.as_str(), moai.matched, moai.complete), ("/rock moai", 1, true));
    }

    #[test]
    fn reads_the_checkout() {
        // The sibling gdev checkout, when this repo has one.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../gdev");
        if !root.join("commands.ts").is_file() {
            return;
        }
        let table = Table::load(&root);
        assert!(table.command("rock").is_some_and(|rock| matches!(rock.argument, Argument::Choice(_))));
        assert!(!table.controls.is_empty());
    }
}
