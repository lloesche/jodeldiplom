//! The prose of a plan as the model writes its JSON ([`Prose`]): fed the
//! answer as it comes, piece by piece, it gives the words worth showing
//! as they come (the story, each character with its looks, what each
//! scene shows, the art's style and why), unescaped, and nothing of the
//! JSON around them. It reads the JSON as it goes, without waiting for
//! it to be whole: which string is a key and which a value, and which key
//! of which field a value is of (serde cannot read a string not yet
//! finished). The decorations around the words are the preparation
//! view's, while its only seam is text (`Story::Words`).

/// The values shown: the field they are in (the key of the object or
/// list they are in, none at the top), their key, and what comes before
/// and after each.
const SHOWN: &[(&str, &str, &str, &str)] = &[
    ("", "story", "", "\n\n"),
    ("characters", "name", "", ": "),
    ("characters", "looks", "", "\n"),
    ("scenes", "shows", "· ", "\n"),
    ("art", "style", "\nThe look: ", "\n"),
    ("art", "why", "", "\n"),
];

/// Where the reader is in the JSON.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum At {
    /// Between strings.
    Between,
    /// In a key.
    Key,
    /// In a value: shown (and what comes after it) or not.
    Value(Option<&'static str>),
}

/// The JSON's prose, read as it comes (see the module's doc).
#[derive(Clone, Debug)]
pub struct Prose {
    at: At,
    /// The containers open: whether each is an object, and the key it is
    /// the value of.
    open: Vec<(bool, String)>,
    /// Whether the next string in the object open is a key.
    want_key: bool,
    /// The key last read, and the one being read.
    key: String,
    /// An escape begun: its text from the backslash.
    escape: Option<String>,
}

impl Default for Prose {
    fn default() -> Prose {
        Prose::new()
    }
}

impl Prose {
    /// Before anything is read.
    pub fn new() -> Prose {
        Prose {
            at: At::Between,
            open: Vec::new(),
            want_key: false,
            key: String::new(),
            escape: None,
        }
    }

    /// The prose in the next piece of the answer, `text`.
    pub fn feed(&mut self, text: &str) -> String {
        let mut out = String::new();
        for c in text.chars() {
            self.read(c, &mut out);
        }
        out
    }

    /// The field the reader is in: the key of the nearest container open
    /// that is a key's value (none at the top).
    fn field(&self) -> &str {
        (self.open.iter().rev())
            .map(|(_, k)| k.as_str())
            .find(|k| !k.is_empty())
            .unwrap_or("")
    }

    fn read(&mut self, c: char, out: &mut String) {
        match self.at {
            At::Between => match c {
                '{' | '[' => {
                    let key = if self.in_object() {
                        std::mem::take(&mut self.key)
                    } else {
                        String::new()
                    };
                    self.open.push((c == '{', key));
                    self.want_key = c == '{';
                }
                '}' | ']' => {
                    self.open.pop();
                    self.want_key = false;
                }
                ',' => self.want_key = self.in_object(),
                ':' => self.want_key = false,
                '"' if self.want_key => {
                    self.at = At::Key;
                    self.key.clear();
                }
                '"' => {
                    let field = self.field();
                    let shown = (SHOWN.iter())
                        .find(|(f, k, _, _)| *f == field && *k == self.key && self.in_object());
                    if let Some((_, _, before, _)) = shown {
                        out.push_str(before);
                    }
                    self.at = At::Value(shown.map(|(_, _, _, after)| *after));
                }
                _ => {}
            },
            At::Key => match self.unescaped(c) {
                Some(None) => {
                    self.at = At::Between;
                    self.want_key = false;
                }
                Some(Some(ch)) => self.key.push(ch),
                None => {}
            },
            At::Value(shown) => match self.unescaped(c) {
                Some(None) => {
                    out.push_str(shown.unwrap_or_default());
                    self.at = At::Between;
                }
                Some(Some(ch)) if shown.is_some() => out.push(ch),
                _ => {}
            },
        }
    }

    /// Whether the container open is an object.
    fn in_object(&self) -> bool {
        self.open.last().is_some_and(|(object, _)| *object)
    }

    /// `c` read inside a string: `Some(None)` its end, `Some(Some(ch))` a
    /// character of it (a line break, a tab or another control character
    /// as a space), `None` part of an escape not done yet (a `\uXXXX`
    /// waits for its pair where it is half of one), or one that stands
    /// for nothing (half a pair alone is dropped, and what follows it read
    /// again as it is).
    fn unescaped(&mut self, c: char) -> Option<Option<char>> {
        let Some(esc) = &mut self.escape else {
            return match c {
                '"' => Some(None),
                '\\' => {
                    self.escape = Some(String::from('\\'));
                    None
                }
                c => Some(Some(c)),
            };
        };
        esc.push(c);
        // Whether it waits for more (`Some(true)`), is whole (`Some(false)`),
        // or is half a pair with something else after it (`None`).
        let waiting = match esc.as_bytes() {
            [b'\\', b'u', hex @ ..] if hex.len() < 4 => Some(true),
            // A high surrogate waits for its low one, as long as one comes.
            [b'\\', b'u', b'd' | b'D', b'8'..=b'9' | b'a'..=b'b' | b'A'..=b'B', _, _, rest @ ..] => {
                let pair = rest.iter().enumerate().all(|(k, &b)| match k {
                    0 => b == b'\\',
                    1 => b == b'u',
                    _ => b.is_ascii_hexdigit(),
                });
                pair.then_some(rest.len() < 6)
            }
            _ => Some(false),
        };
        match waiting {
            Some(true) => return None,
            Some(false) => {}
            None => {
                // What came after the half is read again, as it is.
                let after = esc[6..].to_owned();
                self.escape = None;
                let mut last = None;
                for ch in after.chars() {
                    last = self.unescaped(ch);
                }
                return last;
            }
        }
        let read: Option<String> = serde_json::from_str(&format!("\"{esc}\"")).ok();
        self.escape = None;
        let ch = read?.chars().next()?;
        Some(Some(if ch.is_whitespace() || ch.is_control() {
            ' '
        } else {
            ch
        }))
    }
}
