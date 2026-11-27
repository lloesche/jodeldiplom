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
//! the BPE core's decoding half (`jodel_tensors::bpe::Pieces`).
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

/// What the chat template must write for [`Tokenizer::chat`]'s turns (as
/// the template's text has it).
const TURNS: [&str; 4] = [
    "bos_token",
    "'<|turn>' + role + '\\n'",
    "'<turn|>\\n'",
    "'<|turn>model\\n'",
];

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
        let mut stop = vec![eos];
        for p in STOP_PIECES {
            let i = *ids
                .get(p.as_bytes())
                .ok_or_else(|| format!("no piece {p}"))?;
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
        let mut ids = vec![self.bos];
        ids.extend(self.encode(&format!(
            "<|turn>user\n{}<turn|>\n<|turn>model\n",
            user.trim()
        )));
        ids
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
    /// the stop pieces (`<turn|>` a special); and a few merges.
    const LETTERS: [&str; 11] = [
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

    /// Its tokeniser.
    fn small() -> Tokenizer {
        Tokenizer::new(Parts {
            pieces: pieces(),
            merges: MERGES
                .iter()
                .map(|(a, b)| (a.as_bytes(), b.as_bytes()))
                .collect(),
            specials: vec![0, id("<turn|>")],
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
                i if i == 0 || i == id("<turn|>") as usize => 3,
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
        let template = "{{- bos_token -}}{{- '<|turn>' + role + '\\n' }}{{- '<turn|>\\n' -}}{{- '<|turn>model\\n' -}}";
        let (t, back) = (small(), read(gguf(template)).unwrap());
        for s in ["abc", " abc ab", "<bos>cba é<turn|>"] {
            assert_eq!(back.encode(s), t.encode(s));
        }
        assert_eq!(back.stop(), t.stop());
        assert_eq!(back.chat("ab"), t.chat("ab"));
        let err = read(gguf("{{ messages }}")).err().unwrap();
        assert!(err.contains("chat template"), "{err}");
    }
}
