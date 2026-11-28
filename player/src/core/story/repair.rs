//! JSON read forgivingly ([`json`], [`any`]): a small model's answer,
//! slightly broken, read for everything whole in it, each slip mended and
//! said.
//!
//! - It starts at the answer's first `{` (a model may fence it, or say
//!   something first; [`any`]: or its first `[`, where that comes first)
//!   and ends with that value (anything after it is left).
//! - A trailing comma, or a comma too many, is passed over; a comma
//!   missing between two values is taken as there.
//! - A quote inside a string that does not end it (what follows it is no
//!   colon or bracket, no comma before a key or a value, no key on the
//!   same line) is taken as a quote; a backslash before a
//!   character that is no escape stands for that character; a line break
//!   inside a string stays.
//! - A key or a value written without quotes is taken as a string; a
//!   bracket that closes the wrong container closes the one open.
//! - An answer cut off (its end before its brackets close) keeps all that
//!   is whole: every container open closed, a string or a number cut off
//!   left out, and an object in a list cut off left out whole (a scene
//!   half written is no scene); an object a field is cut off keeps its
//!   fields written whole.
//! - A field written twice keeps its first value.
//! - A container nested deeper than [`MOST_DEPTH`] is left out whole (as
//!   null), read past without going into it: a model caught in a loop of
//!   brackets cannot overflow the stack.

use serde_json::{Map, Number, Value};

/// The deepest a container is read (a plan's answer is 4 deep).
pub const MOST_DEPTH: usize = 32;

/// The object `text` holds from its first `{`, read forgivingly (see the
/// module's doc), each mending said in `notes`; none where it has no `{`.
pub fn json(text: &str, notes: &mut Vec<String>) -> Option<Value> {
    read_from(text, text.find('{')?, notes)
}

/// The object or list `text` holds from its first `{` or `[` (whichever
/// comes first), read as [`json`] reads; none where it has neither.
pub fn any(text: &str, notes: &mut Vec<String>) -> Option<Value> {
    let at = [text.find('{'), text.find('[')]
        .into_iter()
        .flatten()
        .min()?;
    read_from(text, at, notes)
}

/// The value of `text` from its byte `at`, read forgivingly.
fn read_from(text: &str, at: usize, notes: &mut Vec<String>) -> Option<Value> {
    let mut reader = Reader {
        chars: text[at..].chars().collect(),
        at: 0,
        depth: 0,
        notes: Vec::new(),
    };
    let value = match reader.value() {
        Ok(value) => value,
        Err(Cut(part)) => part.unwrap_or(Value::Null),
    };
    notes.append(&mut reader.notes);
    Some(value)
}

/// `value` as text: a string trimmed, a number or a truth as written;
/// none for anything else (or nothing).
pub fn text(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(s) => Some(s.trim().to_owned()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// `value` as a number: a number, or a string that reads as one.
pub fn number(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// `value` as a list: a list's items, one value alone as a list of it,
/// nothing as none.
pub fn list(value: Option<&Value>) -> Vec<Value> {
    match value {
        Some(Value::Array(items)) => items.clone(),
        None | Some(Value::Null) => Vec::new(),
        Some(one) => vec![one.clone()],
    }
}

/// The text ended before a value did: what of it is whole, where that is
/// kept (a container's whole parts), none where nothing is (a string or a
/// number cut off, or an object in a list).
struct Cut(Option<Value>);

/// The answer's characters, where the reader is, how deep in containers,
/// and what it mended.
struct Reader {
    chars: Vec<char>,
    at: usize,
    depth: usize,
    notes: Vec<String>,
}

impl Reader {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.at).copied()
    }

    fn skip_space(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.at += 1;
        }
    }

    /// The first character from `at` that is not a space.
    fn next_from(&self, mut at: usize) -> Option<(usize, char)> {
        while let Some(&c) = self.chars.get(at) {
            if !c.is_whitespace() {
                return Some((at, c));
            }
            at += 1;
        }
        None
    }

    fn note(&mut self, what: &str) {
        let note = format!("the JSON had {what}");
        if !self.notes.contains(&note) {
            self.notes.push(note);
        }
    }

    /// A value from here.
    fn value(&mut self) -> Result<Value, Cut> {
        self.skip_space();
        match self.peek() {
            None => Err(Cut(None)),
            Some('{' | '[') if self.depth >= MOST_DEPTH => self.too_deep(),
            Some(c @ ('{' | '[')) => {
                self.depth += 1;
                let value = if c == '{' {
                    self.object()
                } else {
                    self.array()
                };
                self.depth -= 1;
                value
            }
            Some('"') => self.string().map(Value::String),
            Some(c) if c == '-' || c == '+' || c == '.' || c.is_ascii_digit() => self.number(),
            Some(_) => self.bare(),
        }
    }

    /// A container from its opening bracket, nested too deep: read past
    /// (to its closing bracket, strings stepped over) without going into
    /// it, and taken as null.
    fn too_deep(&mut self) -> Result<Value, Cut> {
        self.note("containers nested too deep: left out");
        let mut open = 0usize;
        let mut in_string = false;
        while let Some(c) = self.peek() {
            self.at += 1;
            match c {
                '\\' if in_string => self.at += 1,
                '"' => in_string = !in_string,
                '{' | '[' if !in_string => open += 1,
                '}' | ']' if !in_string => {
                    open -= 1;
                    if open == 0 {
                        return Ok(Value::Null);
                    }
                }
                _ => {}
            }
        }
        Err(Cut(None))
    }

    /// An object from its `{`.
    fn object(&mut self) -> Result<Value, Cut> {
        self.at += 1;
        let mut map = Map::new();
        loop {
            self.skip_space();
            match self.peek() {
                None => {
                    self.note("its end cut off");
                    return Err(Cut(Some(Value::Object(map))));
                }
                Some('}') => {
                    self.at += 1;
                    return Ok(Value::Object(map));
                }
                Some(']') => {
                    self.note("a bracket closing the wrong container");
                    self.at += 1;
                    return Ok(Value::Object(map));
                }
                Some(',') => {
                    self.note("a comma too many");
                    self.at += 1;
                    continue;
                }
                Some(_) => {}
            }
            let key = match self.key() {
                Ok(key) => key,
                Err(_) => {
                    self.note("its end cut off");
                    return Err(Cut(Some(Value::Object(map))));
                }
            };
            self.skip_space();
            if self.peek() == Some(':') {
                self.at += 1;
            } else {
                self.note("a colon missing");
            }
            let value = match self.value() {
                Ok(value) => value,
                Err(Cut(part)) => {
                    self.note("its end cut off");
                    if let Some(part) = part {
                        map.entry(key).or_insert(part);
                    }
                    return Err(Cut(Some(Value::Object(map))));
                }
            };
            if map.contains_key(&key) {
                self.note(&format!("the field {key:?} twice (the first kept)"));
            } else {
                map.insert(key, value);
            }
            self.after_item('}');
        }
    }

    /// A list from its `[`.
    fn array(&mut self) -> Result<Value, Cut> {
        self.at += 1;
        let mut items = Vec::new();
        loop {
            self.skip_space();
            match self.peek() {
                None => {
                    self.note("its end cut off");
                    return Err(Cut(Some(Value::Array(items))));
                }
                Some(']') => {
                    self.at += 1;
                    return Ok(Value::Array(items));
                }
                Some('}') => {
                    self.note("a bracket closing the wrong container");
                    self.at += 1;
                    return Ok(Value::Array(items));
                }
                Some(',') => {
                    self.note("a comma too many");
                    self.at += 1;
                    continue;
                }
                Some(_) => {}
            }
            let object = self.peek() == Some('{');
            match self.value() {
                Ok(value) => items.push(value),
                Err(Cut(part)) => {
                    self.note("its end cut off");
                    // An object in a list cut off is left out whole.
                    if let Some(part) = part.filter(|_| !object) {
                        items.push(part);
                    }
                    return Err(Cut(Some(Value::Array(items))));
                }
            }
            self.after_item(']');
        }
    }

    /// After an object's field or a list's item: its comma passed (one
    /// before a closing bracket noted, a comma too many), or a comma
    /// missing before the next noted (the closing bracket `close` and the
    /// text's end are left for the container).
    fn after_item(&mut self, close: char) {
        self.skip_space();
        match self.peek() {
            Some(',') => {
                self.at += 1;
                if matches!(self.next_from(self.at), Some((_, '}' | ']'))) {
                    self.note("a comma too many");
                }
            }
            Some(c) if c == close || c == '}' || c == ']' => {}
            None => {}
            Some(_) => self.note("a comma missing"),
        }
    }

    /// A key: a string, or a word up to its colon.
    fn key(&mut self) -> Result<String, Cut> {
        if self.peek() == Some('"') {
            return self.string();
        }
        self.note("a key without quotes");
        let start = self.at;
        while self
            .peek()
            .is_some_and(|c| c != ':' && c != '}' && c != ',')
        {
            self.at += 1;
        }
        if self.peek().is_none() {
            return Err(Cut(None));
        }
        Ok(self.chars[start..self.at]
            .iter()
            .collect::<String>()
            .trim()
            .to_owned())
    }

    /// A string from its opening quote.
    fn string(&mut self) -> Result<String, Cut> {
        self.at += 1;
        let mut s = String::new();
        loop {
            let Some(c) = self.peek() else {
                return Err(Cut(None));
            };
            self.at += 1;
            match c {
                '"' if self.ends_string() => return Ok(s),
                '"' => {
                    self.note("a quote inside a string");
                    s.push('"');
                }
                '\\' => {
                    let Some(e) = self.peek() else {
                        return Err(Cut(None));
                    };
                    self.at += 1;
                    match e {
                        '"' | '\\' | '/' => s.push(e),
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        'r' => s.push('\r'),
                        'b' | 'f' => {}
                        'u' => s.push(self.unicode()?),
                        other => {
                            self.note("a backslash before no escape");
                            s.push(other);
                        }
                    }
                }
                c => s.push(c),
            }
        }
    }

    /// Whether the quote just read ends its string: what follows it is the
    /// text's end, a colon or a bracket; a comma and then what starts a
    /// key or a value (a quote, a bracket, a figure, a word of JSON, or a
    /// word before a colon: a key without quotes); or a quote on a line of
    /// its own, or one that opens a key (`"word":`, a comma missing).
    fn ends_string(&self) -> bool {
        match self.next_from(self.at) {
            None => true,
            Some((_, ':' | '}' | ']')) => true,
            Some((at, '"')) => self.chars[self.at..at].contains(&'\n') || self.key_at(at),
            Some((comma, ',')) => match self.next_from(comma + 1) {
                None => true,
                Some((at, c)) => {
                    let word = |w: &str| {
                        let ahead: String = self.chars[at..].iter().take(w.len()).collect();
                        ahead == w
                    };
                    matches!(c, '"' | '{' | '[' | '}' | ']' | ',' | '-')
                        || c.is_ascii_digit()
                        || word("true")
                        || word("false")
                        || word("null")
                        || self.bare_key_at(at)
                }
            },
            Some(_) => false,
        }
    }

    /// Whether a key in quotes starts at the quote at `at`: its string,
    /// on one line, then a colon.
    fn key_at(&self, at: usize) -> bool {
        let rest = &self.chars[at + 1..];
        let Some(end) = rest.iter().position(|&c| c == '"' || c == '\n') else {
            return false;
        };
        rest[end] == '"' && matches!(self.next_from(at + 2 + end), Some((_, ':')))
    }

    /// Whether a key without quotes starts at `at`: a word, then a colon.
    fn bare_key_at(&self, at: usize) -> bool {
        let rest = &self.chars[at..];
        let end = (rest.iter())
            .position(|c| !(c.is_alphanumeric() || *c == '_'))
            .unwrap_or(rest.len());
        end > 0 && matches!(self.next_from(at + end), Some((_, ':')))
    }

    /// The character of a `\u` escape's four hexadecimal figures (and its
    /// pair's, for half of one); U+FFFD for one that stands for none.
    fn unicode(&mut self) -> Result<char, Cut> {
        let hex = |r: &mut Reader| -> Result<Option<u32>, Cut> {
            if r.at + 4 > r.chars.len() {
                return Err(Cut(None));
            }
            let digits: String = r.chars[r.at..r.at + 4].iter().collect();
            let value = u32::from_str_radix(&digits, 16).ok();
            if value.is_some() {
                r.at += 4;
            }
            Ok(value)
        };
        let Some(first) = hex(self)? else {
            return Ok(char::REPLACEMENT_CHARACTER);
        };
        if (0xD800..0xDC00).contains(&first) {
            let pair =
                self.chars.get(self.at) == Some(&'\\') && self.chars.get(self.at + 1) == Some(&'u');
            if pair {
                self.at += 2;
                if let Some(second) = hex(self)?.filter(|s| (0xDC00..0xE000).contains(s)) {
                    let c = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00);
                    return Ok(char::from_u32(c).unwrap_or(char::REPLACEMENT_CHARACTER));
                }
            }
            return Ok(char::REPLACEMENT_CHARACTER);
        }
        Ok(char::from_u32(first).unwrap_or(char::REPLACEMENT_CHARACTER))
    }

    /// A number: its characters read whole (one that reads as none is
    /// `null`).
    fn number(&mut self) -> Result<Value, Cut> {
        let start = self.at;
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_digit() || "+-.eE".contains(c))
        {
            self.at += 1;
        }
        if self.peek().is_none() {
            return Err(Cut(None));
        }
        let text: String = self.chars[start..self.at].iter().collect();
        let text = text.strip_prefix('+').unwrap_or(&text);
        if let Ok(n) = text.parse::<i64>() {
            return Ok(Value::Number(n.into()));
        }
        match text.parse::<f64>().ok().and_then(Number::from_f64) {
            Some(n) => Ok(Value::Number(n)),
            None => {
                self.note("a number that is none");
                Ok(Value::Null)
            }
        }
    }

    /// A word without quotes: `true`, `false`, `null` (in any case,
    /// `None` too), else a string up to the next comma or bracket.
    fn bare(&mut self) -> Result<Value, Cut> {
        let start = self.at;
        while self.peek().is_some_and(|c| !",}]\n".contains(c)) {
            self.at += 1;
        }
        if self.peek().is_none() {
            return Err(Cut(None));
        }
        let word: String = self.chars[start..self.at].iter().collect();
        let word = word.trim();
        Ok(match word.to_ascii_lowercase().as_str() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            "null" | "none" => Value::Null,
            _ => {
                self.note("a value without quotes");
                Value::String(word.to_owned())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `text` read, and what was mended.
    fn read(text: &str) -> (Value, Vec<String>) {
        let mut notes = Vec::new();
        let value = json(text, &mut notes).expect("an object");
        (value, notes)
    }

    /// Good JSON reads as serde reads it, nothing mended; what comes
    /// before its first brace and after its end is left.
    #[test]
    fn good_json_reads_as_it_is() {
        let text =
            r#"{"a": [1, -2.5, "x\"y\né🚗"], "b": {"c": null, "d": true, "e": false}, "f": 1e3}"#;
        let want: Value = serde_json::from_str(text).unwrap();
        let (got, notes) = read(&format!("```json\n{text}\n```\nthat's all {{"));
        assert_eq!(got, want);
        assert!(notes.is_empty(), "{notes:?}");
        assert!(json("no object", &mut Vec::new()).is_none());
    }

    /// Each slip a small model makes, mended and said.
    #[test]
    fn slips_are_mended_and_said() {
        let cases: [(&str, Value, &str); 11] = [
            (
                r#"{"a": [1, 2,], "b": 3,}"#,
                json!({"a": [1, 2], "b": 3}),
                "a comma too many",
            ),
            (
                r#"{"a": 1 "b": 2}"#,
                json!({"a": 1, "b": 2}),
                "a comma missing",
            ),
            (
                r#"{"shows": "He says "go", and leaves", "n": 1}"#,
                json!({"shows": "He says \"go\", and leaves", "n": 1}),
                "a quote inside a string",
            ),
            (
                r#"{"s": "a\'b", "t": "line one
line two"}"#,
                json!({"s": "a'b", "t": "line one\nline two"}),
                "a backslash before no escape",
            ),
            (
                r#"{shot: wide, "n": True}"#,
                json!({"shot": "wide", "n": true}),
                "a key without quotes",
            ),
            (
                r#"{"a": [1, 2}, "b": 3]"#,
                json!({"a": [1, 2], "b": 3}),
                "a bracket closing the wrong container",
            ),
            (
                r#"{"a": 1, "a": 2}"#,
                json!({"a": 1}),
                "the field \"a\" twice (the first kept)",
            ),
            (
                r#"{"x": NaN-ish, "y": 2}"#,
                json!({"x": "NaN-ish", "y": 2}),
                "a value without quotes",
            ),
            (
                r#"{"x": 1.2.3, "y": 2}"#,
                json!({"x": null, "y": 2}),
                "a number that is none",
            ),
            (
                r#"{"a": "x", b: 1, "c": 2}"#,
                json!({"a": "x", "b": 1, "c": 2}),
                "a key without quotes",
            ),
            (
                r#"{"a": "x" "b": 2, "c": 3}"#,
                json!({"a": "x", "b": 2, "c": 3}),
                "a comma missing",
            ),
        ];
        for (text, want, said) in cases {
            let (got, notes) = read(text);
            assert_eq!(got, want, "{text}");
            assert!(
                notes.iter().any(|n| n.contains(said)),
                "{said:?} not in {notes:?}"
            );
        }
    }

    /// How deep `value`'s containers go.
    fn depth(value: &Value) -> usize {
        match value {
            Value::Array(items) => 1 + items.iter().map(depth).max().unwrap_or(0),
            Value::Object(fields) => 1 + fields.values().map(depth).max().unwrap_or(0),
            _ => 0,
        }
    }

    /// Containers nested past the most are left out whole, read past
    /// without going into them, closed or cut off; a list read from its
    /// bracket where that comes first.
    #[test]
    fn containers_nested_too_deep_are_left_out() {
        for (open, close) in [("[", "]"), ("{\"a\": ", "}")] {
            let deep = format!(
                "{{\"x\": {}1{}, \"y\": 2}}",
                open.repeat(100_000),
                close.repeat(100_000)
            );
            let (got, notes) = read(&deep);
            assert_eq!(got["y"], json!(2));
            assert!(depth(&got) <= MOST_DEPTH, "{} deep", depth(&got));
            assert!(
                notes.iter().any(|n| n.contains("nested too deep")),
                "{notes:?}"
            );
            let cut = format!("{{\"y\": 2, \"x\": {}", open.repeat(100_000));
            let (got, _) = read(&cut);
            assert!(got["y"] == json!(2) && depth(&got) <= MOST_DEPTH);
        }
        let mut notes = Vec::new();
        assert_eq!(
            any("[1, {\"a\": 2}]", &mut notes),
            Some(json!([1, {"a": 2}]))
        );
        assert_eq!(any("so {\"a\": [1]}", &mut notes), Some(json!({"a": [1]})));
    }

    /// An answer cut off keeps everything whole: the scenes written whole
    /// (the one cut off left out), the fields of an object cut off written
    /// whole, nothing of a string or a number cut off.
    #[test]
    fn an_answer_cut_off_keeps_all_that_is_whole() {
        let text = r#"{"story": "s", "scenes": [{"first_line": 0, "prompt": "a"}, {"first_line": 4, "prompt": "b"}, {"first_line": 9, "pro"#;
        let (got, notes) = read(text);
        assert_eq!(
            got,
            json!({"story": "s", "scenes": [{"first_line": 0, "prompt": "a"}, {"first_line": 4, "prompt": "b"}]})
        );
        assert!(notes.iter().any(|n| n.contains("its end cut off")));
        let (got, _) = read(r#"{"art": {"genre": "folk", "era": "9"#);
        assert_eq!(got, json!({"art": {"genre": "folk"}}));
        let (got, _) = read(r#"{"a": 12"#);
        assert_eq!(got, json!({}));
        let (got, _) = read(r#"{"motifs": ["rain", "boats", "li"#);
        assert_eq!(got, json!({"motifs": ["rain", "boats"]}));
        let (got, _) = read("{");
        assert_eq!(got, json!({}));
    }
}
