//! A small JSON reader, enough for a web service's answers and a
//! checkpoint's header: objects, arrays, strings (with `\u` escapes and
//! surrogate pairs), numbers, `true`, `false` and `null`.

/// A JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number.
    Number(f64),
    /// A string.
    String(String),
    /// An array.
    Array(Vec<Json>),
    /// An object, its members in order.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// The value `text` holds (white space around it allowed), if it is
    /// JSON.
    pub fn parse(text: &str) -> Option<Json> {
        let mut reader = Reader {
            bytes: text.as_bytes(),
            at: 0,
            depth: 0,
        };
        let value = reader.value()?;
        reader.space();
        (reader.at == reader.bytes.len()).then_some(value)
    }

    /// The member `key` of an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The string, if it is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    /// The number, if it is one.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// The truth value, if it is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The items, if it is an array.
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }
}

/// How deep arrays and objects may nest.
const DEEPEST: usize = 128;

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    depth: usize,
}

impl Reader<'_> {
    fn space(&mut self) {
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
        {
            self.at += 1;
        }
    }

    fn eat(&mut self, b: u8) -> bool {
        self.space();
        if self.bytes.get(self.at) == Some(&b) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn word(&mut self, word: &str, value: Json) -> Option<Json> {
        let end = self.at + word.len();
        (self.bytes.get(self.at..end) == Some(word.as_bytes())).then(|| {
            self.at = end;
            value
        })
    }

    fn value(&mut self) -> Option<Json> {
        self.space();
        match *self.bytes.get(self.at)? {
            b'{' => self.nested(Self::object),
            b'[' => self.nested(Self::array),
            b'"' => self.string().map(Json::String),
            b't' => self.word("true", Json::Bool(true)),
            b'f' => self.word("false", Json::Bool(false)),
            b'n' => self.word("null", Json::Null),
            _ => self.number(),
        }
    }

    fn nested(&mut self, read: fn(&mut Self) -> Option<Json>) -> Option<Json> {
        self.depth += 1;
        if self.depth > DEEPEST {
            return None;
        }
        let value = read(self);
        self.depth -= 1;
        value
    }

    fn object(&mut self) -> Option<Json> {
        self.at += 1;
        let mut members = Vec::new();
        if self.eat(b'}') {
            return Some(Json::Object(members));
        }
        loop {
            self.space();
            let key = self.string()?;
            if !self.eat(b':') {
                return None;
            }
            members.push((key, self.value()?));
            if self.eat(b'}') {
                return Some(Json::Object(members));
            }
            if !self.eat(b',') {
                return None;
            }
        }
    }

    fn array(&mut self) -> Option<Json> {
        self.at += 1;
        let mut items = Vec::new();
        if self.eat(b']') {
            return Some(Json::Array(items));
        }
        loop {
            items.push(self.value()?);
            if self.eat(b']') {
                return Some(Json::Array(items));
            }
            if !self.eat(b',') {
                return None;
            }
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        let digits = std::str::from_utf8(self.bytes.get(self.at..self.at + 4)?).ok()?;
        self.at += 4;
        u32::from_str_radix(digits, 16).ok()
    }

    fn string(&mut self) -> Option<String> {
        if self.bytes.get(self.at) != Some(&b'"') {
            return None;
        }
        self.at += 1;
        let mut out = Vec::new();
        loop {
            let b = *self.bytes.get(self.at)?;
            self.at += 1;
            match b {
                b'"' => return String::from_utf8(out).ok(),
                b'\\' => {
                    let e = *self.bytes.get(self.at)?;
                    self.at += 1;
                    let c = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let high = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&high) {
                                if self.bytes.get(self.at..self.at + 2) != Some(b"\\u") {
                                    return None;
                                }
                                self.at += 2;
                                let low = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return None;
                                }
                                0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
                            } else {
                                high
                            };
                            char::from_u32(code)?
                        }
                        _ => return None,
                    };
                    let mut buffer = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buffer).as_bytes());
                }
                0..0x20 => return None,
                _ => out.push(b),
            }
        }
    }

    fn number(&mut self) -> Option<Json> {
        let start = self.at;
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| matches!(b, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'))
        {
            self.at += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).ok()?;
        if text.is_empty() {
            return None;
        }
        text.parse().ok().map(Json::Number)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_are_read() {
        let v = Json::parse(
            r#" {"id": 42, "name": "Caf\u00e9 \"Ole\" \ud83c\udfb5", "ok": true,
                "none": null, "list": [1.5, -2e3, false], "empty": {}} "#,
        )
        .unwrap();
        assert_eq!(v.get("id").and_then(Json::as_f64), Some(42.0));
        assert_eq!(
            v.get("name").and_then(Json::as_str),
            Some("Café \"Ole\" 🎵")
        );
        assert_eq!(v.get("ok").and_then(Json::as_bool), Some(true));
        assert_eq!(v.get("none"), Some(&Json::Null));
        assert_eq!(
            v.get("list").and_then(Json::as_array),
            Some(&[Json::Number(1.5), Json::Number(-2000.0), Json::Bool(false)][..])
        );
        assert_eq!(v.get("empty"), Some(&Json::Object(Vec::new())));
        assert_eq!(v.get("missing"), None);
        assert_eq!(Json::parse("[]"), Some(Json::Array(Vec::new())));
        assert_eq!(Json::parse("\"a\\nb\""), Some(Json::String("a\nb".into())));
    }

    #[test]
    fn broken_text_is_not_json() {
        for text in [
            "",
            "{",
            "[1,]",
            "{\"a\" 1}",
            "\"open",
            "tru",
            "[1] 2",
            "\"\\x\"",
            "\"\\ud800\"",
        ] {
            assert_eq!(Json::parse(text), None, "{text}");
        }
        assert_eq!(Json::parse(&"[".repeat(500)), None);
    }
}
