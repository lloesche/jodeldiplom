//! The text encoder's tokens ([`Tokenizer`]): Qwen's byte-level BPE
//! (`vocab.json`, `merges.txt`, the added tokens of
//! `tokenizer_config.json`), the prompt in the model's chat template, as
//! FLUX.2 klein's pipeline makes it: a user turn, the assistant's turn
//! begun with an empty thought, padded with `<|endoftext|>` to 512 tokens.
//!
//! The text is split as Qwen2's pre-tokenizer's pattern splits it
//! ([`pieces`]), its letters taken as Unicode's alphabetic characters
//! (which also counts a few combining marks as letters), and not
//! normalized first (the tokenizer's NFC): prompts in composed form, as
//! typed, come out the same.

use std::collections::HashMap;
use std::path::Path;

use jodel_tensors::bpe::{Merges, Part, Specials};
use jodel_tensors::json::Json;

/// The tokens a prompt is padded to.
pub const TOKENS: usize = 512;
/// `<|endoftext|>`: the padding.
const PAD: u32 = 151_643;

/// The tokenizer's files in the checkpoint's folder: its vocabulary, its
/// merges, its configuration (with the added tokens).
pub(crate) const VOCAB_FILE: &str = "vocab.json";
pub(crate) const MERGES_FILE: &str = "merges.txt";
pub(crate) const CONFIG_FILE: &str = "tokenizer_config.json";

/// Qwen's tokenizer.
pub struct Tokenizer {
    /// Each byte's id: its character in GPT-2's table (each byte as a
    /// printable character) as a piece.
    bytes: [u32; 256],
    merges: Merges,
    /// The added tokens.
    added: Specials,
}

impl Tokenizer {
    /// The tokenizer in `dir` (Qwen's `vocab.json`, `merges.txt`,
    /// `tokenizer_config.json`).
    pub fn open(dir: &Path) -> Result<Tokenizer, String> {
        let read = |name: &str| {
            std::fs::read_to_string(dir.join(name))
                .map_err(|e| format!("{}: {e}", dir.join(name).display()))
        };
        let Some(Json::Object(vocab)) = Json::parse(&read(VOCAB_FILE)?) else {
            return Err("vocab.json is not an object".into());
        };
        let ids = vocab
            .into_iter()
            .map(|(k, v)| Some((k.into_bytes(), v.as_f64()? as u32)))
            .collect::<Option<HashMap<_, _>>>()
            .ok_or("vocab.json has a token without an id")?;
        let mut pieces = vec![Vec::new(); ids.values().max().map_or(0, |&m| m as usize + 1)];
        for (piece, &id) in &ids {
            pieces[id as usize].clone_from(piece);
        }
        let id = |s: &str| {
            ids.get(s.as_bytes())
                .copied()
                .ok_or_else(|| format!("no token for {s:?}"))
        };
        let pairs = read(MERGES_FILE)?
            .lines()
            .filter(|l| !l.starts_with("#version") && !l.is_empty())
            .map(|l| {
                let (a, b) = l.split_once(' ').ok_or("a merge that is not a pair")?;
                Ok((id(a)?, id(b)?))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let merges = Merges::new(&pieces, &ids, &pairs)?;
        let chars = byte_chars();
        let mut bytes = [0; 256];
        for (b, slot) in bytes.iter_mut().enumerate() {
            *slot = id(&chars[b].to_string())?;
        }
        let config =
            Json::parse(&read(CONFIG_FILE)?).ok_or("tokenizer_config.json does not parse")?;
        let Some(Json::Object(decoder)) = config.get("added_tokens_decoder") else {
            return Err("tokenizer_config.json has no added tokens".into());
        };
        let added = decoder
            .iter()
            .map(|(id, t)| {
                let content = t.get("content").and_then(Json::as_str);
                match (content, id.parse()) {
                    (Some(c), Ok(id)) => Ok((c.as_bytes().to_vec(), id)),
                    _ => Err(format!("an added token {id} without its content")),
                }
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Tokenizer {
            bytes,
            merges,
            added: Specials::new(added),
        })
    }

    /// `text`'s tokens: the added tokens where they are, the rest split
    /// into pieces ([`pieces`]) and each merged by BPE.
    pub fn tokens(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for part in self.added.split(text) {
            match part {
                Part::Text(t) => {
                    for piece in pieces(t) {
                        self.bpe(piece, &mut out);
                    }
                }
                Part::Special(id) => out.push(id),
            }
        }
        out
    }

    /// `prompt` as the text encoder takes it: in the chat template, padded
    /// to [`TOKENS`] (cut there if longer); the tokens, and how many are
    /// the prompt's (the rest padding).
    pub fn prompt(&self, prompt: &str) -> (Vec<u32>, usize) {
        let text = format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let mut ids = self.tokens(&text);
        ids.truncate(TOKENS);
        let real = ids.len();
        ids.resize(TOKENS, PAD);
        (ids, real)
    }

    /// One piece's tokens, by BPE over its bytes' characters
    /// (`jodel_tensors::bpe`).
    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        let ids = piece.bytes().map(|b| self.bytes[usize::from(b)]).collect();
        self.merges.word(ids, out);
    }
}

/// GPT-2's table: printable Latin-1 bytes as themselves, the rest from
/// U+0100 on, in order.
fn byte_chars() -> [char; 256] {
    let printable =
        |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    let mut out = ['\0'; 256];
    let mut next = 256;
    for b in 0..256u32 {
        out[b as usize] = if printable(b) {
            char::from_u32(b).unwrap_or('\0')
        } else {
            next += 1;
            char::from_u32(next - 1).unwrap_or('\0')
        };
    }
    out
}

fn letter(c: char) -> bool {
    c.is_alphabetic() && !c.is_numeric()
}

fn newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// `text` split as Qwen2's pre-tokenizer's pattern splits it (each
/// alternative tried in order at each place):
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
/// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`.
pub(crate) fn pieces(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let at = |i: usize| chars.get(i).map(|&(_, c)| c);
    let byte = |i: usize| chars.get(i).map_or(text.len(), |&(b, _)| b);
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i].1;
        let end = (|| {
            // Contractions.
            if c == '\'' {
                for s in ["s", "t", "re", "ve", "m", "ll", "d"] {
                    let n = s.len();
                    let matched = (0..n).all(|j| {
                        at(i + 1 + j)
                            .is_some_and(|x| x.to_ascii_lowercase() == s.as_bytes()[j] as char)
                    });
                    if matched {
                        return i + 1 + n;
                    }
                }
            }
            // Letters, with one other character before them.
            let first = if letter(c) {
                Some(i)
            } else if !newline(c) && !c.is_numeric() && at(i + 1).is_some_and(letter) {
                Some(i + 1)
            } else {
                None
            };
            if let Some(mut j) = first {
                while at(j).is_some_and(letter) {
                    j += 1;
                }
                return j;
            }
            if c.is_numeric() {
                return i + 1;
            }
            // Other characters, with a space before them, then newlines.
            let other = |x: char| !x.is_whitespace() && !letter(x) && !x.is_numeric();
            let start = if c == ' ' && at(i + 1).is_some_and(other) {
                i + 1
            } else {
                i
            };
            if at(start).is_some_and(other) {
                let mut j = start;
                while at(j).is_some_and(other) {
                    j += 1;
                }
                while at(j).is_some_and(newline) {
                    j += 1;
                }
                return j;
            }
            // Whitespace: up to its last newline; else all of it at the
            // end, or all but its last before something else.
            let mut run = i;
            while at(run).is_some_and(char::is_whitespace) {
                run += 1;
            }
            if let Some(last) = (i..run).rev().find(|&j| newline(chars[j].1)) {
                return last + 1;
            }
            if run == chars.len() || run - i == 1 {
                run
            } else {
                run - 1
            }
        })();
        out.push(&text[byte(i)..byte(end)]);
        i = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::pieces;

    #[test]
    fn text_splits_as_qwens_pattern_splits_it() {
        assert_eq!(pieces("Hello world"), ["Hello", " world"]);
        assert_eq!(pieces("it's we'LL"), ["it", "'s", " we", "'LL"]);
        assert_eq!(
            pieces("a.b, c!? (x)"),
            ["a", ".b", ",", " c", "!?", " (", "x", ")"]
        );
        assert_eq!(pieces("123 4"), ["1", "2", "3", " ", "4"]);
        assert_eq!(pieces("  two  spaces"), [" ", " two", " ", " spaces"]);
        assert_eq!(pieces("end   "), ["end", "   "]);
        assert_eq!(pieces("a\n\nb"), ["a", "\n\n", "b"]);
        assert_eq!(pieces("x  \n  y"), ["x", "  \n", " ", " y"]);
        assert_eq!(pieces("ok.\n"), ["ok", ".\n"]);
        assert_eq!(pieces("café 東京"), ["café", " 東京"]);
    }
}
