//! JSON as JavaScript's `JSON.parse` reads it, which gdev's parsers relied
//! on: numbers are the nearest f64; an object's later duplicate key replaces
//! the earlier one's value in its place; and an object's keys come in
//! JavaScript's property order, the array indices ("0", "1"...) first in
//! numeric order, then the rest as written. A lone UTF-16 surrogate, which
//! Rust strings cannot hold, becomes U+FFFD; everywhere gdev's parsers look,
//! it is a character the font lacks either way.

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    /// JavaScript's `value[key]` on an object, or none.
    pub(crate) fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(fields) => fields.iter().find(|(name, _)| name == key).map(|(_, value)| value),
            _ => None,
        }
    }
}

/// The JSON value `text` holds, or why it does not hold one, in V8's words.
pub(crate) fn parse(text: &str) -> Result<Json, String> {
    let mut reader = Reader { text, offset: 0 };
    reader.whitespace();
    let value = reader.value(0)?;
    reader.whitespace();
    match reader.peek() {
        None => Ok(value),
        Some(_) => Err(reader.at("Unexpected non-whitespace character after JSON")),
    }
}

/// How deeply arrays and objects may nest.
const DEPTH: usize = 512;

struct Reader<'a> {
    text: &'a str,
    offset: usize,
}

impl Reader<'_> {
    fn peek(&self) -> Option<char> {
        self.text[self.offset..].chars().next()
    }

    /// `problem` where the reader is, as V8 places it: in UTF-16 code units,
    /// with `\r`, `\n` and `\r\n` each ending a line.
    fn at(&self, problem: &str) -> String {
        let before = &self.text[..self.offset];
        let position = before.encode_utf16().count();
        let (mut line, mut column) = (1, 1);
        let mut characters = before.chars().peekable();
        while let Some(character) = characters.next() {
            if character == '\r' && characters.peek() == Some(&'\n') {
                continue;
            }
            if character == '\r' || character == '\n' {
                line += 1;
                column = 1;
            } else {
                column += character.len_utf16();
            }
        }
        format!("{problem} at position {position} (line {line} column {column})")
    }

    /// The character where the reader is, unexpected, with the text around
    /// it as V8 quotes it: all of a short text, else ten code units each way.
    fn unexpected(&self) -> String {
        let Some(token) = self.peek() else {
            return "Unexpected end of JSON input".to_string();
        };
        let units: Vec<u16> = self.text.encode_utf16().collect();
        let position = self.text[..self.offset].encode_utf16().count();
        let quote = |range: std::ops::Range<usize>| String::from_utf16_lossy(&units[range]);
        let context = if units.len() <= 20 {
            format!("\"{}\"", self.text)
        } else if position < 10 {
            format!("\"{}\"...", quote(0..position + 10))
        } else if position < units.len() - 10 {
            format!("...\"{}\"...", quote(position - 10..position + 10))
        } else {
            format!("...\"{}\"", quote(position - 10..units.len()))
        };
        format!("Unexpected token '{token}', {context} is not valid JSON")
    }

    fn whitespace(&mut self) {
        while let Some(' ' | '\t' | '\n' | '\r') = self.peek() {
            self.offset += 1;
        }
    }

    fn eat(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.offset += expected.len_utf8();
            true
        } else {
            false
        }
    }

    fn digits(&mut self) -> bool {
        let start = self.offset;
        while let Some('0'..='9') = self.peek() {
            self.offset += 1;
        }
        self.offset > start
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > DEPTH {
            return Err(self.at("JSON nests too deeply"));
        }
        match self.peek() {
            Some('{') => self.object(depth),
            Some('[') => self.array(depth),
            Some('"') => self.string().map(Json::String),
            Some('-' | '0'..='9') => self.number(),
            Some(first) => {
                let words = [("true", Json::Bool(true)), ("false", Json::Bool(false)), ("null", Json::Null)];
                let Some((word, value)) = words.into_iter().find(|(word, _)| word.starts_with(first)) else {
                    return Err(self.unexpected());
                };
                for expected in word.chars() {
                    if !self.eat(expected) {
                        return Err(self.unexpected());
                    }
                }
                Ok(value)
            }
            None => Err(self.unexpected()),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, String> {
        self.offset += 1;
        let mut fields: Vec<(String, Json)> = Vec::new();
        self.whitespace();
        if !self.eat('}') {
            loop {
                if self.peek() != Some('"') {
                    return Err(self.at(if fields.is_empty() {
                        "Expected property name or '}' in JSON"
                    } else {
                        "Expected double-quoted property name in JSON"
                    }));
                }
                let key = self.string()?;
                self.whitespace();
                if !self.eat(':') {
                    return Err(self.at("Expected ':' after property name in JSON"));
                }
                self.whitespace();
                let value = self.value(depth + 1)?;
                match fields.iter_mut().find(|(name, _)| *name == key) {
                    Some((_, earlier)) => *earlier = value,
                    None => fields.push((key, value)),
                }
                self.whitespace();
                if self.eat('}') {
                    break;
                }
                if !self.eat(',') {
                    return Err(self.at("Expected ',' or '}' after property value in JSON"));
                }
                self.whitespace();
            }
        }
        order_keys(&mut fields);
        Ok(Json::Object(fields))
    }

    fn array(&mut self, depth: usize) -> Result<Json, String> {
        self.offset += 1;
        let mut items = Vec::new();
        self.whitespace();
        if !self.eat(']') {
            loop {
                items.push(self.value(depth + 1)?);
                self.whitespace();
                if self.eat(']') {
                    break;
                }
                if !self.eat(',') {
                    return Err(self.at("Expected ',' or ']' after array element in JSON"));
                }
                self.whitespace();
            }
        }
        Ok(Json::Array(items))
    }

    fn string(&mut self) -> Result<String, String> {
        self.offset += 1;
        let mut string = String::new();
        loop {
            match self.peek() {
                None => return Err(self.at("Unterminated string in JSON")),
                Some('"') => {
                    self.offset += 1;
                    return Ok(string);
                }
                Some('\0'..='\u{1f}') => return Err(self.at("Bad control character in string literal in JSON")),
                Some('\\') => {
                    self.offset += 1;
                    let escaped = match self.peek() {
                        Some('"') => '"',
                        Some('\\') => '\\',
                        Some('/') => '/',
                        Some('b') => '\u{8}',
                        Some('f') => '\u{c}',
                        Some('n') => '\n',
                        Some('r') => '\r',
                        Some('t') => '\t',
                        Some('u') => {
                            self.offset += 1;
                            string.push(self.unicode_escape()?);
                            continue;
                        }
                        Some(_) => return Err(self.at("Bad escaped character in JSON")),
                        None => return Err(self.unexpected()),
                    };
                    self.offset += 1;
                    string.push(escaped);
                }
                Some(next) => {
                    self.offset += next.len_utf8();
                    string.push(next);
                }
            }
        }
    }

    /// The character of a `\u` escape, after the `u`: a surrogate pair's two
    /// escapes make one, and a lone surrogate is U+FFFD.
    fn unicode_escape(&mut self) -> Result<char, String> {
        let unit = self.hex4()?;
        if (0xd800..0xdc00).contains(&unit) && self.text[self.offset..].starts_with("\\u") {
            let saved = self.offset;
            self.offset += 2;
            let low = self.hex4()?;
            if (0xdc00..0xe000).contains(&low) {
                let point = 0x10000 + ((unit - 0xd800) << 10) + (low - 0xdc00);
                return Ok(char::from_u32(point).unwrap_or('\u{fffd}'));
            }
            self.offset = saved;
        }
        Ok(char::from_u32(unit).unwrap_or('\u{fffd}'))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let mut unit = 0;
        for _ in 0..4 {
            let Some(digit) = self.peek().and_then(|digit| digit.to_digit(16)) else {
                return Err(self.at("Bad Unicode escape in JSON"));
            };
            unit = unit * 16 + digit;
            self.offset += 1;
        }
        Ok(unit)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.offset;
        if self.eat('-') && !matches!(self.peek(), Some('0'..='9')) {
            return Err(self.at("No number after minus sign in JSON"));
        }
        if self.eat('0') {
            if let Some('0'..='9') = self.peek() {
                return Err(self.at("Unexpected number in JSON"));
            }
        } else {
            self.digits();
        }
        if self.eat('.') && !self.digits() {
            return Err(self.at("Unterminated fractional number in JSON"));
        }
        if self.eat('e') || self.eat('E') {
            let _ = self.eat('+') || self.eat('-');
            if !self.digits() {
                return Err(self.at("Exponent part is missing a number in JSON"));
            }
        }
        // Rust's parse is correctly rounded, as JavaScript's is; too big a
        // number is an infinity in both.
        let number = self.text[start..self.offset].parse().map_err(|_| self.at("Bad number in JSON"))?;
        Ok(Json::Number(number))
    }
}

/// Puts an object's keys in JavaScript's property order: array indices first,
/// by number, then the rest in the order they came.
pub(crate) fn order_keys<T>(fields: &mut [(String, T)]) {
    fields.sort_by_key(|(key, _)| array_index(key).map_or((1, 0), |index| (0, index)));
}

/// The array index a property key is in JavaScript, if it is one: a
/// canonical decimal below 2³² − 1.
fn array_index(key: &str) -> Option<u32> {
    let canonical =
        key == "0" || (key.starts_with(|c: char| ('1'..='9').contains(&c)) && key.bytes().all(|b| b.is_ascii_digit()));
    key.parse::<u32>().ok().filter(|index| canonical && *index != u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(fields: &[(&str, Json)]) -> Json {
        Json::Object(fields.iter().map(|(key, value)| (key.to_string(), value.clone())).collect())
    }

    #[test]
    fn reads_values() {
        let text = r#" {"a": [1, -2.5e3, true, false, null], "b": "x\"\\\/\b\f\n\r\t\u00e5\ud83d\ude00\ud800"} "#;
        assert_eq!(
            parse(text).unwrap(),
            object(&[
                (
                    "a",
                    Json::Array(vec![
                        Json::Number(1.0),
                        Json::Number(-2500.0),
                        Json::Bool(true),
                        Json::Bool(false),
                        Json::Null
                    ])
                ),
                ("b", Json::String("x\"\\/\u{8}\u{c}\n\r\tå😀\u{fffd}".to_string())),
            ])
        );
        assert_eq!(parse("0.1").unwrap(), Json::Number(0.1));
        assert_eq!(parse("1e400").unwrap(), Json::Number(f64::INFINITY));
        assert_eq!(parse("123456789012345678901234567890").unwrap(), Json::Number(1.2345678901234568e29));
    }

    #[test]
    fn orders_keys_as_javascript_does() {
        let parsed = parse(r#"{"b": 1, "10": 2, "a": 3, "2": 4, "b": 5, "01": 6, "4294967295": 7}"#).unwrap();
        let keys: Vec<&str> = match &parsed {
            Json::Object(fields) => fields.iter().map(|(key, _)| key.as_str()).collect(),
            _ => unreachable!(),
        };
        assert_eq!(keys, ["2", "10", "b", "a", "01", "4294967295"]);
        assert_eq!(parsed.get("b"), Some(&Json::Number(5.0)));
    }

    #[test]
    fn refuses_what_is_not_json() {
        for text in ["", "{", "{a: 1}", "[1,]", "[1 2]", "01", "1.", "-", "1e", "\"\u{1}\"", "\"\\x\"", "nul", "{} x"] {
            assert!(parse(text).is_err(), "{text:?}");
        }
    }

    #[test]
    fn says_what_is_wrong_as_v8_does() {
        // What Deno's JSON.parse says of each.
        let cases = [
            ("", "Unexpected end of JSON input"),
            ("{ meshes: [] }", "Expected property name or '}' in JSON at position 2 (line 1 column 3)"),
            ("{\"a\": 1,}", "Expected double-quoted property name in JSON at position 8 (line 1 column 9)"),
            ("[1,]", "Unexpected token ']', \"[1,]\" is not valid JSON"),
            ("[1 2]", "Expected ',' or ']' after array element in JSON at position 3 (line 1 column 4)"),
            ("{\"a\" 1}", "Expected ':' after property name in JSON at position 5 (line 1 column 6)"),
            ("{\"a\": 1 \"b\": 2}", "Expected ',' or '}' after property value in JSON at position 8 (line 1 column 9)"),
            ("01", "Unexpected number in JSON at position 1 (line 1 column 2)"),
            ("1.", "Unterminated fractional number in JSON at position 2 (line 1 column 3)"),
            ("[-]", "No number after minus sign in JSON at position 2 (line 1 column 3)"),
            ("1.5ex", "Exponent part is missing a number in JSON at position 4 (line 1 column 5)"),
            ("\"\u{1}\"", "Bad control character in string literal in JSON at position 1 (line 1 column 2)"),
            ("\"\\x\"", "Bad escaped character in JSON at position 2 (line 1 column 3)"),
            ("\"\\u12G4\"", "Bad Unicode escape in JSON at position 5 (line 1 column 6)"),
            ("\"\\u12", "Bad Unicode escape in JSON at position 5 (line 1 column 6)"),
            ("\"ab\\", "Unexpected end of JSON input"),
            ("nul", "Unexpected end of JSON input"),
            ("nulx", "Unexpected token 'x', \"nulx\" is not valid JSON"),
            ("{} x", "Unexpected non-whitespace character after JSON at position 3 (line 1 column 4)"),
            ("\"abc", "Unterminated string in JSON at position 4 (line 1 column 5)"),
            ("{\"a\":1", "Expected ',' or '}' after property value in JSON at position 6 (line 1 column 7)"),
            ("\"😀\" x", "Unexpected non-whitespace character after JSON at position 5 (line 1 column 6)"),
            ("[1,\r\n2,\r3\n x]", "Expected ',' or ']' after array element in JSON at position 11 (line 4 column 2)"),
            ("[x                  ]", "Unexpected token 'x', \"[x         \"... is not valid JSON"),
            (
                "[1,1,1,1,1,x,2,2,2,2,2,2,2,2,2,2,2,2]",
                "Unexpected token 'x', ...\"1,1,1,1,1,x,2,2,2,2,\"... is not valid JSON",
            ),
            ("                         x     ", "Unexpected token 'x', ...\"          x     \" is not valid JSON"),
        ];
        for (text, message) in cases {
            assert_eq!(parse(text).unwrap_err(), message, "{text:?}");
        }
    }
}
