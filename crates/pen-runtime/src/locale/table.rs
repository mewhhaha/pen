//! String tables: the game's `text.json` and its translations, a JSON object
//! from key to a string, to an object of plural forms keyed by CLDR's
//! category names (`"other"` required), or to a list of strings reached as
//! `key.0`, `key.1`, ...
use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, bail};
use pen_protocol::plural_category::{self, OTHER};
use serde_json::Value;

/// A table's string for a key.
#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    Text(String),
    /// A form for each plural category the table gives, by category code
    /// (`pen_protocol::plural_category`); `other`, the last, is always there.
    Plural([Option<String>; 6]),
}

impl Entry {
    /// The string for a count of plural `category`: a plural's form for it,
    /// or its `other` form when it has none.
    pub fn form(&self, category: u32) -> &str {
        match self {
            Entry::Text(text) => text,
            Entry::Plural(forms) => {
                let form = |category: u32| forms.get(category as usize).and_then(Option::as_deref);
                form(category).or_else(|| form(OTHER)).unwrap_or_default()
            }
        }
    }

    /// The plural categories it has a form for: just `other` for a string.
    pub fn categories(&self) -> Vec<u32> {
        match self {
            Entry::Text(_) => vec![OTHER],
            Entry::Plural(forms) => (0..).zip(forms).filter(|(_, form)| form.is_some()).map(|(code, _)| code).collect(),
        }
    }

    /// Every string it holds.
    pub fn strings(&self) -> Vec<&str> {
        match self {
            Entry::Text(text) => vec![text],
            Entry::Plural(forms) => forms.iter().flatten().map(String::as_str).collect(),
        }
    }

    /// The placeholders of all its strings.
    pub fn placeholders(&self) -> BTreeSet<&str> {
        self.strings().into_iter().flat_map(placeholders).collect()
    }
}

/// A string table, by key.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Table {
    entries: BTreeMap<String, Entry>,
}

impl Table {
    /// Reads a table from JSON text. A value that is no string, object of
    /// plural forms or list of strings is an error naming its key, as is a
    /// plural object without `other` or with a name that is no category.
    pub fn parse(text: &str) -> anyhow::Result<Table> {
        let value: Value = serde_json::from_str(text).map_err(|error| anyhow!("not JSON: {error}"))?;
        let Value::Object(object) = value else {
            bail!("a string table is a JSON object from key to string, not {}", kind(&value));
        };
        let mut table = Table::default();
        for (key, value) in object {
            match value {
                Value::String(text) => table.insert(key, Entry::Text(text))?,
                Value::Array(list) => {
                    for (index, value) in list.into_iter().enumerate() {
                        let Value::String(text) = value else {
                            bail!("\"{key}\": a list holds strings, not {}", kind(&value));
                        };
                        table.insert(format!("{key}.{index}"), Entry::Text(text))?;
                    }
                }
                Value::Object(forms) => {
                    let mut plural: [Option<String>; 6] = Default::default();
                    for (name, value) in forms {
                        let Some(category) = plural_category::NAMES.iter().position(|known| *known == name) else {
                            bail!(
                                "\"{key}\": \"{name}\" is no plural category ({})",
                                plural_category::NAMES.join(", ")
                            );
                        };
                        let Value::String(text) = value else {
                            bail!("\"{key}\": the \"{name}\" form is a string, not {}", kind(&value));
                        };
                        plural[category] = Some(text);
                    }
                    if plural[OTHER as usize].is_none() {
                        bail!("\"{key}\": a plural object needs an \"other\" form");
                    }
                    table.insert(key, Entry::Plural(plural))?;
                }
                value => {
                    bail!(
                        "\"{key}\": a value is a string, an object of plural forms or a list of strings, not {}",
                        kind(&value)
                    )
                }
            }
        }
        Ok(table)
    }

    /// Adds an entry; a key given twice (`list.0` beside a list) is an error.
    fn insert(&mut self, key: String, entry: Entry) -> anyhow::Result<()> {
        if self.entries.contains_key(&key) {
            bail!("\"{key}\" is given twice");
        }
        self.entries.insert(key, entry);
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<&Entry> {
        self.entries.get(key)
    }

    /// Its entries, in key order.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &Entry)> {
        self.entries.iter().map(|(key, entry)| (key.as_str(), entry))
    }
}

/// What a JSON value is, for an error.
fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
    }
}

/// The names of a string's placeholders: `{count}`, `{name}`; letters,
/// digits and underscores between braces.
fn placeholders(text: &str) -> impl Iterator<Item = &str> {
    text.split('{').skip(1).filter_map(|after| {
        let (name, _) = after.split_once('}')?;
        (!name.is_empty() && name.chars().all(|char| char.is_alphanumeric() || char == '_')).then_some(name)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_protocol::plural_category::{FEW, ONE};

    #[test]
    fn tables_hold_strings_plural_forms_and_lists() {
        let table = Table::parse(
            r#"{"title": "Gnorp", "coins": {"one": "{count} coin", "other": "{count} coins"}, "swears": ["Fan!", "Attans!"]}"#,
        )
        .unwrap();
        assert_eq!(table.get("title"), Some(&Entry::Text("Gnorp".into())));
        let coins = table.get("coins").unwrap();
        assert_eq!(coins.form(ONE), "{count} coin");
        assert_eq!(coins.form(FEW), "{count} coins");
        assert_eq!(coins.categories(), [ONE, OTHER]);
        assert_eq!(table.get("swears.1"), Some(&Entry::Text("Attans!".into())));
        assert_eq!(table.get("swears"), None);
        assert_eq!(table.entries().map(|(key, _)| key).collect::<Vec<_>>(), ["coins", "swears.0", "swears.1", "title"]);
    }

    #[test]
    fn bad_tables_say_which_key_is_wrong() {
        let error = |text: &str| Table::parse(text).unwrap_err().to_string();
        assert_eq!(
            error(r#"{"speed": 3}"#),
            "\"speed\": a value is a string, an object of plural forms or a list of strings, not a number"
        );
        assert_eq!(error(r#"{"coins": {"one": "a coin"}}"#), "\"coins\": a plural object needs an \"other\" form");
        assert_eq!(
            error(r#"{"coins": {"several": "coins", "other": "coins"}}"#),
            "\"coins\": \"several\" is no plural category (zero, one, two, few, many, other)"
        );
        assert_eq!(error(r#"{"lines": ["a", 2]}"#), "\"lines\": a list holds strings, not a number");
        assert_eq!(error(r#"{"lines": ["a"], "lines.0": "b"}"#), "\"lines.0\" is given twice");
        assert_eq!(error("[]"), "a string table is a JSON object from key to string, not a list");
        assert_eq!(error("{"), "not JSON: EOF while parsing an object at line 1 column 1");
    }

    #[test]
    fn placeholders_are_names_in_braces() {
        assert_eq!(placeholders("{count} of {total_2}, {} { x} {").collect::<Vec<_>>(), ["count", "total_2"]);
    }
}
