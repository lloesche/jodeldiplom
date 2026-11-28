//! Gemma's tokeniser: byte-pair encoding over its 262 144 pieces, as the
//! GGUF's metadata defines it (the pieces, their types, the merges).
//!
//! - The text is cut at the special tokens (the pieces of the control and
//!   user-defined types: `<bos>`, `<|turn>`, …), the longest that matches
//!   first; each special is its own id.
//! - Between them, every space becomes `▁`, and the whole stretch is one
//!   word (the checkpoint's split on spaces finds none after that).
//! - The word starts as its characters (a character with no piece of its
//!   own as the pieces of its UTF-8 bytes, `<0x41>`), then the pair of
//!   neighbours with the lowest merge rank is merged, the leftmost first,
//!   until no pair merges (`jodel_tensors::bpe`): what Hugging Face's
//!   `tokenizers` does.
//!
//! Decoding turns `▁` back into spaces and byte pieces into their bytes,
//! the specials left out: each piece's text worked out once, read back by
//! the BPE core's decoding half (`jodel_tensors::bpe::Pieces`). A thinking
//! answer's thought is told apart from the answer by its specials
//! ([`Thinking`]), as the decoding leaves them out.
use std::collections::HashMap;

use jodel_tensors::bpe::{Merges, Part, Pieces, Specials};
use jodel_tensors::gguf::{Array, Gguf};

/// The pieces that end the model's answer, with the file's end-of-sequence
/// token (the generation config's: the end of a turn, `<eos>`, a tool's
/// answer begun).
const STOP_PIECES: [&str; 3] = ["<eos>", "<turn|>", "<|tool_response>"];

/// The pieces' types that are special tokens (GGUF's control and
/// user-defined).
const CONTROL: i64 = 3;
const USER_DEFINED: i64 = 4;

/// What the chat template must write for [`Tokenizer::chat`]'s turns, the
/// system turn of [`Tokenizer::chat_thinking`] (its thinking switched on)
/// and the thought an answer begins with ([`Thinking`]), as the template's
/// text has it.
const TURNS: [&str; 8] = [
    "bos_token",
    "'<|turn>' + role + '\\n'",
    "'<turn|>\\n'",
    "'<|turn>model\\n'",
    "'<|turn>system\\n'",
    "'<|think|>\\n'",
    "'<|channel>thought\\n'",
    "'\\n<channel|>'",
];

/// The specials a thought is between.
const THOUGHT_PIECES: [&str; 2] = ["<|channel>", "<channel|>"];

/// The tokeniser.
pub struct Tokenizer {
    /// Each id's text: its piece's bytes with `▁` a space, a byte piece's
    /// byte, none for a special.
    text: Pieces,
    /// Each piece's id, by its bytes (`▁` as its UTF-8).
    ids: HashMap<Vec<u8>, u32>,
    /// Each mergeable pair of ids: its rank and the id it makes.
    merges: Merges,
    specials: Specials,
    /// The piece of each byte (`<0x00>` …).
    bytes: [u32; 256],
    /// The token a prompt begins with.
    bos: u32,
    /// The tokens that end an answer.
    stop: Vec<u32>,
    /// The specials a thought is between.
    thought: [u32; 2],
}

/// `▁`, which stands for a space in the pieces.
const SPACE: &str = "\u{2581}";

/// What a tokeniser is made of: its pieces, its merges (pairs of pieces,
/// in rank order), its special tokens' ids, the token a prompt begins
/// with, the one the file says ends one (with [`STOP_PIECES`], those that
/// end an answer).
struct Parts<'a> {
    pieces: Vec<Vec<u8>>,
    merges: Vec<(&'a [u8], &'a [u8])>,
    specials: Vec<u32>,
    bos: u32,
    eos: u32,
}

impl Tokenizer {
    /// The tokeniser of a GGUF's metadata (`tokenizer.ggml.*`: Gemma 4's
    /// pieces, their types, the merges in rank order as `"a b"`, the
    /// beginning and end tokens), its chat template checked to write the
    /// turns as [`Tokenizer::chat`] does.
    pub fn from_gguf<R>(file: &Gguf<R>) -> Result<Tokenizer, String> {
        let model = file.get("tokenizer.ggml.model")?.as_str();
        if model != Some("gemma4") {
            return Err(format!("a tokeniser of {model:?}, not gemma4"));
        }
        let list = |key: &str| -> Result<&Array, String> {
            file.get(key)?
                .as_array()
                .ok_or_else(|| format!("{key} is not a list"))
        };
        let strings = |key: &str| -> Result<&[String], String> {
            list(key)?
                .strings()
                .ok_or_else(|| format!("{key} is not of strings"))
        };
        let id = |key: &str| -> Result<u32, String> {
            file.get(key)?
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| format!("{key} is not an id"))
        };
        let tokens = strings("tokenizer.ggml.tokens")?;
        let pieces: Vec<Vec<u8>> = tokens.iter().map(|s| s.as_bytes().to_vec()).collect();
        let merges = strings("tokenizer.ggml.merges")?
            .iter()
            .map(|m| {
                m.split_once(' ')
                    .map(|(a, b)| (a.as_bytes(), b.as_bytes()))
                    .ok_or_else(|| format!("a merge that is not a pair: {m}"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let types = list("tokenizer.ggml.token_type")?
            .ints()
            .ok_or("tokenizer.ggml.token_type is not of whole numbers")?;
        if types.len() != pieces.len() {
            return Err("a type for each piece, not as many".into());
        }
        let specials = (0..)
            .zip(&types)
            .filter(|(_, t)| matches!(**t, CONTROL | USER_DEFINED))
            .map(|(i, _)| i)
            .collect();
        let template = file.get("tokenizer.chat_template")?.as_str().unwrap_or("");
        if let Some(missing) = TURNS.iter().find(|t| !template.contains(*t)) {
            return Err(format!("a chat template that does not write {missing}"));
        }
        Tokenizer::new(Parts {
            pieces,
            merges,
            specials,
            bos: id("tokenizer.ggml.bos_token_id")?,
            eos: id("tokenizer.ggml.eos_token_id")?,
        })
    }

    /// From its parts: an error where a merge is not of two pieces making
    /// a third, or a byte (`<0x41>`) or a stop piece has no piece.
    fn new(parts: Parts) -> Result<Tokenizer, String> {
        let Parts {
            pieces,
            merges,
            specials,
            bos,
            eos,
        } = parts;
        let ids: HashMap<Vec<u8>, u32> = (0..)
            .zip(&pieces)
            .map(|(i, p): (u32, &Vec<u8>)| (p.clone(), i))
            .collect();
        let piece = |p: &str| {
            ids.get(p.as_bytes())
                .copied()
                .ok_or_else(|| format!("no piece {p}"))
        };
        let thought = [piece(THOUGHT_PIECES[0])?, piece(THOUGHT_PIECES[1])?];
        let mut stop = vec![eos];
        for p in STOP_PIECES {
            let i = piece(p)?;
            if !stop.contains(&i) {
                stop.push(i);
            }
        }
        let pairs = merges
            .iter()
            .map(|&(a, b)| match (ids.get(a), ids.get(b)) {
                (Some(&x), Some(&y)) => Ok((x, y)),
                _ => Err(format!(
                    "a merge of pieces not in the vocabulary: {} {}",
                    String::from_utf8_lossy(a),
                    String::from_utf8_lossy(b)
                )),
            })
            .collect::<Result<Vec<_>, String>>()?;
        let merges = Merges::new(&pieces, &ids, &pairs)?;
        let mut text = Pieces::default();
        for (i, piece) in pieces.iter().enumerate() {
            let bytes = match byte_of(piece) {
                Some(b) => vec![b],
                None => match std::str::from_utf8(piece) {
                    Ok(s) => s.replace(SPACE, " ").into_bytes(),
                    Err(_) => piece.clone(),
                },
            };
            text.put(i, bytes, false);
        }
        for &i in &specials {
            text.put(i as usize, pieces[i as usize].clone(), true);
        }
        let specials = Specials::new(
            specials
                .iter()
                .map(|&i| (pieces[i as usize].clone(), i))
                .collect(),
        );
        let mut bytes = [0; 256];
        for (b, slot) in bytes.iter_mut().enumerate() {
            let name = format!("<0x{b:02X}>");
            *slot = *ids
                .get(name.as_bytes())
                .ok_or_else(|| format!("no piece {name}"))?;
        }
        Ok(Tokenizer {
            text,
            ids,
            merges,
            specials,
            bytes,
            bos,
            stop,
            thought,
        })
    }

    /// The number of pieces.
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// Whether it has no pieces.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// The tokens that end an answer.
    pub fn stop(&self) -> &[u32] {
        &self.stop
    }

    /// The ids of `text`: its special tokens as they are, the rest
    /// byte-pair encoded.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for part in self.specials.split(text) {
            match part {
                Part::Text(t) => self.word(t, &mut out),
                Part::Special(id) => out.push(id),
            }
        }
        out
    }

    /// The ids of a stretch of text between specials (one word).
    fn word(&self, text: &str, out: &mut Vec<u32>) {
        let text = text.replace(' ', SPACE);
        let mut ids: Vec<u32> = Vec::with_capacity(text.len());
        let mut buf = [0u8; 4];
        for c in text.chars() {
            let s = c.encode_utf8(&mut buf);
            match self.ids.get(s.as_bytes()) {
                Some(&id) => ids.push(id),
                None => ids.extend(s.bytes().map(|b| self.bytes[usize::from(b)])),
            }
        }
        self.merges.word(ids, out);
    }

    /// The text of `ids`, the special tokens left out.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.text.decode(ids)
    }

    /// A decoder of tokens as they come ([`Stream`]).
    pub fn stream(&self) -> Stream<'_> {
        Stream {
            tokens: self,
            held: Vec::new(),
        }
    }

    /// The prompt of one user turn as the model's chat template writes it
    /// (no thinking): the beginning token, the turn, then the model's turn
    /// opened.
    pub fn chat(&self, user: &str) -> Vec<u32> {
        self.turns("", user)
    }

    /// [`Tokenizer::chat`]'s prompt with the model's thinking switched on,
    /// as the template writes it (`enable_thinking`): a system turn of the
    /// thinking token only before the user's. The answer begins with its
    /// thought ([`Tokenizer::thinking`]).
    pub fn chat_thinking(&self, user: &str) -> Vec<u32> {
        self.turns("<|turn>system\n<|think|>\n<turn|>\n", user)
    }

    /// The user's next turn after the model's answer, as the chat template
    /// writes it: the model's turn ended (its stop token, never among the
    /// tokens written: [`Tokenizer::stop`]), the user's turn, then the
    /// model's opened again (thinking as the conversation began:
    /// [`crate::Gemma::go_on`] goes on with it).
    pub fn next_turn(&self, user: &str) -> Vec<u32> {
        self.encode(&format!(
            "<turn|>\n<|turn>user\n{}<turn|>\n<|turn>model\n",
            user.trim()
        ))
    }

    /// The beginning token, `system`, the user's turn, the model's opened.
    fn turns(&self, system: &str, user: &str) -> Vec<u32> {
        let mut ids = vec![self.bos];
        ids.extend(self.encode(&format!(
            "{system}<|turn>user\n{}<turn|>\n<|turn>model\n",
            user.trim()
        )));
        ids
    }

    /// The answer's tokens told apart as they come, the thought it begins
    /// with (if any) of at most `thought` tokens, then the answer of at
    /// most `answer` ([`Thinking`]).
    pub fn thinking(&self, thought: usize, answer: usize) -> Thinking {
        Thinking {
            marks: self.thought,
            rooms: [thought, answer],
            taken: [0, 0],
            last: None,
            over: false,
        }
    }
}

/// What a token of an answer is ([`Thinking::take`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Take {
    /// The special that opens the thought, the answer's first.
    Opened,
    /// A token of the thought (its channel's name, `thought`, the first).
    Thought,
    /// The special that closes it.
    Closed,
    /// A token of the answer.
    Answer,
    /// One past the thought's room or the answer's: it is cut off there
    /// (every token after too).
    Over,
}

/// An answer's tokens told apart as they come ([`Tokenizer::thinking`]):
/// those between the thought's specials, where the answer begins with
/// its opening one, and the answer after it (all of it, where it begins
/// with none); each in its room, the first past it over.
#[derive(Clone, Debug)]
pub struct Thinking {
    marks: [u32; 2],
    /// The thought's room and the answer's, and the tokens each took.
    rooms: [usize; 2],
    taken: [usize; 2],
    /// The last token taken in its rooms.
    last: Option<Take>,
    over: bool,
}

impl Thinking {
    /// What token `id`, the next, is.
    pub fn take(&mut self, id: u32) -> Take {
        if self.over {
            return Take::Over;
        }
        let [open, close] = self.marks;
        let (take, room) = match self.last {
            None if id == open => (Take::Opened, None),
            _ if self.in_thought() && id == close => (Take::Closed, None),
            _ if self.in_thought() => (Take::Thought, Some(0)),
            _ => (Take::Answer, Some(1)),
        };
        if let Some(r) = room {
            if self.taken[r] == self.rooms[r] {
                self.over = true;
                return Take::Over;
            }
            self.taken[r] += 1;
        }
        self.last = Some(take);
        take
    }

    /// Whether the thought is under way (opened, not closed: where the
    /// answer ends there, it was cut off in its thought).
    pub fn in_thought(&self) -> bool {
        matches!(self.last, Some(Take::Opened | Take::Thought))
    }

    /// The thought's tokens taken (between its specials), and the
    /// answer's.
    pub fn taken(&self) -> [usize; 2] {
        self.taken
    }

    /// The tokens a generation may write for it at most: both rooms, the
    /// thought's two specials, and one past to find it over.
    pub fn most(&self) -> usize {
        self.rooms[0] + self.rooms[1] + 3
    }

    /// Whether the answer ended by itself where its generation `stopped`
    /// on a stop token: within its room, its thought closed.
    pub fn ended(&self, stopped: bool) -> bool {
        stopped && !self.over && !self.in_thought()
    }
}

/// Tokens decoded as they come ([`Tokenizer::stream`]): each token's bytes
/// decoded once, held until they make whole characters (a character can
/// come in byte pieces over several tokens); what it gives, put together,
/// is [`Tokenizer::decode`]'s text of all of them.
pub struct Stream<'a> {
    tokens: &'a Tokenizer,
    /// The bytes of a character not whole yet.
    held: Vec<u8>,
}

impl Stream<'_> {
    /// The text that token `id` completes (empty while a character is
    /// still in pieces; a byte that starts none as U+FFFD).
    pub fn push(&mut self, id: u32) -> String {
        self.held.extend_from_slice(self.tokens.text.text(id));
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.held) {
                Ok(s) => {
                    out.push_str(s);
                    self.held.clear();
                    return out;
                }
                Err(e) => {
                    let (whole, rest) = self.held.split_at(e.valid_up_to());
                    out.push_str(std::str::from_utf8(whole).unwrap_or_default());
                    let Some(bad) = e.error_len() else {
                        // A character begun: kept for the next token.
                        self.held = rest.to_vec();
                        return out;
                    };
                    out.push(char::REPLACEMENT_CHARACTER);
                    self.held = rest[bad..].to_vec();
                }
            }
        }
    }

    /// What is left once the tokens end (a character never finished, as
    /// U+FFFD).
    pub fn finish(self) -> String {
        String::from_utf8_lossy(&self.held).into_owned()
    }
}

/// The byte a byte piece (`<0x41>`) stands for.
fn byte_of(piece: &[u8]) -> Option<u8> {
    let hex = piece.strip_prefix(b"<0x")?.strip_suffix(b">")?;
    (hex.len() == 2)
        .then(|| u8::from_str_radix(std::str::from_utf8(hex).ok()?, 16).ok())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small vocabulary's pieces: `<bos>` (0), the bytes, letters, `▁`,
    /// the stop pieces (`<turn|>` a special), the thought's specials; and
    /// a few merges.
    const LETTERS: [&str; 13] = [
        "a",
        "b",
        "c",
        SPACE,
        "ab",
        "abc",
        "\u{2581}a",
        "bc",
        "<eos>",
        "<turn|>",
        "<|tool_response>",
        "<|channel>",
        "<channel|>",
    ];
    const MERGES: [(&str, &str); 4] = [("a", "b"), ("ab", "c"), (SPACE, "a"), ("b", "c")];

    fn pieces() -> Vec<Vec<u8>> {
        let mut pieces: Vec<Vec<u8>> = vec![b"<bos>".to_vec()];
        pieces.extend((0..=255u8).map(|b| format!("<0x{b:02X}>").into_bytes()));
        pieces.extend(LETTERS.iter().map(|p| p.as_bytes().to_vec()));
        pieces
    }

    /// The id of piece `p` there.
    fn id(p: &str) -> u32 {
        pieces().iter().position(|q| q == p.as_bytes()).unwrap() as u32
    }

    /// Its specials.
    fn specials() -> Vec<u32> {
        vec![0, id("<turn|>"), id("<|channel>"), id("<channel|>")]
    }

    /// Its tokeniser.
    fn small() -> Tokenizer {
        Tokenizer::new(Parts {
            pieces: pieces(),
            merges: MERGES
                .iter()
                .map(|(a, b)| (a.as_bytes(), b.as_bytes()))
                .collect(),
            specials: specials(),
            bos: 0,
            eos: id("<turn|>"),
        })
        .unwrap()
    }

    #[test]
    fn words_merge_by_rank_and_bytes_fall_back() {
        let t = small();
        // "abc": ab first (rank 0), then abc (rank 1).
        assert_eq!(t.encode("abc"), vec![id("abc")]);
        // " abc": ▁a (rank 2) waits for ab (rank 0): ▁, abc.
        assert_eq!(t.encode(" abc"), vec![id(SPACE), id("abc")]);
        // "bc" alone merges; "é" is two byte pieces.
        assert_eq!(t.encode("bcé"), vec![id("bc"), 1 + 0xC3, 1 + 0xA9]);
        assert_eq!(t.encode("<bos>ab"), vec![0, id("ab")]);
        assert_eq!(t.decode(&t.encode("<bos> abc bcé")), " abc bcé");
        // The beginning token by id, the turn trimmed; `<turn|>` a special.
        let chat = t.chat(" ab ");
        assert_eq!(chat[0], 0);
        assert_eq!(
            chat[1..],
            t.encode("<|turn>user\nab<turn|>\n<|turn>model\n")
        );
        assert!(chat.contains(&id("<turn|>")));
        let thinking = t.chat_thinking(" ab ");
        assert_eq!(thinking[0], 0);
        assert_eq!(
            thinking[1..],
            t.encode("<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nab<turn|>\n<|turn>model\n")
        );
    }

    /// A thought is what is between its specials at the answer's start;
    /// an answer that begins with none is all answer, its specials too.
    /// Each fills its room exactly, the next token over (and all after);
    /// the answer ended by itself only on a stop, within its room, its
    /// thought closed.
    #[test]
    fn a_thought_is_told_apart_from_the_answer_each_in_its_room() {
        use Take::{Answer, Closed, Opened, Over, Thought};
        let t = small();
        let (open, close, a) = (id("<|channel>"), id("<channel|>"), id("a"));
        let takes = |thinking: &mut Thinking, ids: &[u32]| -> Vec<Take> {
            ids.iter().map(|&i| thinking.take(i)).collect()
        };
        // Both rooms filled exactly, a stop after: ended.
        let mut both = t.thinking(2, 3);
        let ids = [open, a, a, close, a, open, close];
        assert_eq!(
            takes(&mut both, &ids),
            [Opened, Thought, Thought, Closed, Answer, Answer, Answer]
        );
        assert_eq!(both.taken(), [2, 3]);
        assert!(!both.in_thought() && both.ended(true));
        assert!(!both.ended(false), "at a limit");
        assert_eq!(both.most(), 2 + 3 + 3);
        // A thought one past its room: over, cut off in its thought.
        let mut long = t.thinking(2, 3);
        assert_eq!(
            takes(&mut long, &[open, a, a, a, close]),
            [Opened, Thought, Thought, Over, Over]
        );
        assert!(long.in_thought() && !long.ended(true));
        // An answer one past its room: over.
        let mut wordy = t.thinking(2, 1);
        assert_eq!(
            takes(&mut wordy, &[open, close, a, a]),
            [Opened, Closed, Answer, Over]
        );
        assert!(!wordy.ended(true));
        // A thought never closed (a stop inside it): cut off.
        let mut cut = t.thinking(4, 3);
        takes(&mut cut, &[open, a]);
        assert!(cut.in_thought() && !cut.ended(true), "never closed");
        // No thought: all answer, its specials too.
        let mut none = t.thinking(0, 3);
        assert_eq!(
            takes(&mut none, &[a, close, open]),
            [Answer, Answer, Answer]
        );
        assert_eq!(none.taken(), [0, 3]);
        assert!(none.ended(true));
    }

    /// A character in byte pieces over several tokens comes out whole once
    /// its last piece is in, a byte that starts none as U+FFFD, and the
    /// pieces together are the whole decode.
    #[test]
    fn a_stream_gives_whole_characters() {
        let t = small();
        let byte = |b: u32| 1 + b;
        let ids = [
            id("a"),
            byte(0xC3),
            byte(0xA9),
            byte(0xFF),
            id("bc"),
            byte(0xC3),
            id(SPACE),
            byte(0xE6),
            byte(0x9D),
        ];
        let mut stream = t.stream();
        let pieces: Vec<String> = ids.iter().map(|&i| stream.push(i)).collect();
        let last = stream.finish();
        assert_eq!(
            pieces,
            ["a", "", "é", "\u{FFFD}", "bc", "", "\u{FFFD} ", "", ""]
        );
        assert_eq!(last, "\u{FFFD}");
        assert_eq!(pieces.concat() + &last, t.decode(&ids));
    }

    /// A GGUF of only `small`'s tokeniser (its pieces' types: control for
    /// `<bos>`, bytes, normal) and a chat template of `template`.
    fn gguf(template: &str) -> Vec<u8> {
        fn string(out: &mut Vec<u8>, s: &[u8]) {
            out.extend((s.len() as u64).to_le_bytes());
            out.extend(s);
        }
        let pieces = pieces();
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
        out.extend(7u64.to_le_bytes());
        string(&mut out, b"tokenizer.ggml.model");
        out.extend(8u32.to_le_bytes());
        string(&mut out, b"gemma4");
        string(&mut out, b"tokenizer.ggml.tokens");
        out.extend(9u32.to_le_bytes());
        out.extend(8u32.to_le_bytes());
        out.extend((pieces.len() as u64).to_le_bytes());
        for p in &pieces {
            string(&mut out, p);
        }
        string(&mut out, b"tokenizer.ggml.token_type");
        out.extend(9u32.to_le_bytes());
        out.extend(5u32.to_le_bytes());
        out.extend((pieces.len() as u64).to_le_bytes());
        for i in 0..pieces.len() {
            let kind: i32 = match i {
                i if specials().contains(&(i as u32)) => 3,
                1..=256 => 6,
                _ => 1,
            };
            out.extend(kind.to_le_bytes());
        }
        string(&mut out, b"tokenizer.ggml.merges");
        out.extend(9u32.to_le_bytes());
        out.extend(8u32.to_le_bytes());
        out.extend((MERGES.len() as u64).to_le_bytes());
        for (a, b) in MERGES {
            let pair = format!("{a} {b}");
            string(&mut out, pair.as_bytes());
        }
        string(&mut out, b"tokenizer.chat_template");
        out.extend(8u32.to_le_bytes());
        string(&mut out, template.as_bytes());
        for (key, value) in [
            ("tokenizer.ggml.bos_token_id", 0),
            ("tokenizer.ggml.eos_token_id", id("<turn|>")),
        ] {
            string(&mut out, key.as_bytes());
            out.extend(4u32.to_le_bytes());
            out.extend(value.to_le_bytes());
        }
        out
    }

    fn read(bytes: Vec<u8>) -> Result<Tokenizer, String> {
        let len = bytes.len() as u64;
        Tokenizer::from_gguf(&Gguf::from_reader(std::io::Cursor::new(bytes), len)?)
    }

    #[test]
    fn it_reads_from_a_gguf() {
        let template = "{{- bos_token -}}{{- '<|turn>' + role + '\\n' }}{{- '<turn|>\\n' -}}{{- '<|turn>model\\n' -}}{{- '<|turn>system\\n' -}}{{- '<|think|>\\n' -}}{{- '<|channel>thought\\n' + text + '\\n<channel|>' -}}";
        let (t, back) = (small(), read(gguf(template)).unwrap());
        for s in ["abc", " abc ab", "<bos>cba é<turn|>"] {
            assert_eq!(back.encode(s), t.encode(s));
        }
        assert_eq!(back.stop(), t.stop());
        assert_eq!(back.chat("ab"), t.chat("ab"));
        assert_eq!(back.chat_thinking("ab"), t.chat_thinking("ab"));
        let err = read(gguf("{{ messages }}")).err().unwrap();
        assert!(err.contains("chat template"), "{err}");
    }
}
