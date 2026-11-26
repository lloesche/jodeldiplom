//! The recogniser's tokens as text: Qwen's byte-level BPE vocabulary
//! (`vocab.json`, each token's bytes spelt with printable characters) and
//! its added tokens (`tokenizer_config.json`), for reading what the model
//! writes. Only decoding: the prompts are fixed token ids.

use std::path::Path;

use jodel_tensors::json::Json;

/// `<|endoftext|>`.
pub const END_OF_TEXT: u32 = 151_643;
/// `<|im_start|>`.
pub const IM_START: u32 = 151_644;
/// `<|im_end|>`.
pub const IM_END: u32 = 151_645;
/// `<|audio_start|>`.
pub const AUDIO_START: u32 = 151_669;
/// `<|audio_end|>`.
pub const AUDIO_END: u32 = 151_670;
/// `<|audio_pad|>`: where the audio's embeddings go.
pub const AUDIO_PAD: u32 = 151_676;
/// `<asr_text>`: between the language and the words.
pub const ASR_TEXT: u32 = 151_704;
/// `system`, `user`, `assistant`, and a newline.
const SYSTEM: u32 = 8948;
const USER: u32 = 872;
const ASSISTANT: u32 = 77091;
const NEWLINE: u32 = 198;

/// The prompt around `audio` audio tokens: an empty system turn, the
/// audio as the user's, and the start of the assistant's (as the model's
/// chat template writes it).
pub fn prompt(audio: usize) -> Vec<u32> {
    let mut ids = vec![
        IM_START,
        SYSTEM,
        NEWLINE,
        IM_END,
        NEWLINE,
        IM_START,
        USER,
        NEWLINE,
        AUDIO_START,
    ];
    ids.extend(std::iter::repeat_n(AUDIO_PAD, audio));
    ids.extend([AUDIO_END, IM_END, NEWLINE, IM_START, ASSISTANT, NEWLINE]);
    ids
}

/// Whether the model has finished.
pub fn is_end(id: u32) -> bool {
    id == IM_END || id == END_OF_TEXT
}

/// The byte each printable character of the vocabulary stands for (GPT-2's
/// table: printable Latin-1 bytes as themselves, the rest from U+0100 on).
fn byte_of_char() -> std::collections::HashMap<char, u8> {
    let mut printable: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut chars: Vec<u32> = printable.clone();
    let mut next = 0;
    for b in 0..=255u32 {
        if !printable.contains(&b) {
            printable.push(b);
            chars.push(256 + next);
            next += 1;
        }
    }
    printable
        .iter()
        .zip(&chars)
        .filter_map(|(&b, &c)| Some((char::from_u32(c)?, b as u8)))
        .collect()
}

/// The vocabulary: each token's bytes, and whether it is special (left out
/// of text).
#[derive(Clone, Debug)]
pub struct Vocab {
    bytes: Vec<Vec<u8>>,
    special: Vec<bool>,
}

impl Vocab {
    /// From the files `vocab.json` and `tokenizer_config.json` in `dir`.
    pub fn open(dir: &Path) -> Result<Vocab, String> {
        let read = |name: &str| {
            std::fs::read_to_string(dir.join(name)).map_err(|e| format!("{name}: {e}"))
        };
        Vocab::parse(&read("vocab.json")?, &read("tokenizer_config.json")?)
    }

    /// From `vocab.json` and `tokenizer_config.json`'s text.
    pub fn parse(vocab_json: &str, config_json: &str) -> Result<Vocab, String> {
        let table = byte_of_char();
        let Some(Json::Object(entries)) = Json::parse(vocab_json) else {
            return Err("vocab.json is not an object".into());
        };
        let mut bytes: Vec<Vec<u8>> = Vec::new();
        let mut special = Vec::new();
        let mut put = |id: usize, b: Vec<u8>, s: bool| {
            if bytes.len() <= id {
                bytes.resize(id + 1, Vec::new());
                special.resize(id + 1, false);
            }
            bytes[id] = b;
            special[id] = s;
        };
        for (token, id) in &entries {
            let id = id.as_f64().ok_or("a token id is not a number")? as usize;
            let b: Option<Vec<u8>> = token.chars().map(|c| table.get(&c).copied()).collect();
            put(
                id,
                b.ok_or(format!("token {id} has a character outside the table"))?,
                false,
            );
        }
        let config = Json::parse(config_json).ok_or("tokenizer_config.json is broken")?;
        if let Some(Json::Object(added)) = config.get("added_tokens_decoder") {
            for (id, t) in added {
                let id: usize = id.parse().map_err(|_| "an added token's id")?;
                let content = t.get("content").and_then(Json::as_str).unwrap_or("");
                let s = t.get("special").and_then(Json::as_bool).unwrap_or(false);
                put(id, content.as_bytes().to_vec(), s);
            }
        }
        Ok(Vocab { bytes, special })
    }

    /// As bytes: for each token its length (u16), whether it is special
    /// (u8), and its bytes. [`Vocab::from_bytes`] reads them back.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (b, &s) in self.bytes.iter().zip(&self.special) {
            out.extend_from_slice(&(b.len() as u16).to_le_bytes());
            out.push(u8::from(s));
            out.extend_from_slice(b);
        }
        out
    }

    /// From [`Vocab::to_bytes`]'s.
    pub fn from_bytes(data: &[u8]) -> Result<Vocab, String> {
        let (mut bytes, mut special) = (Vec::new(), Vec::new());
        let mut at = 0;
        while at < data.len() {
            let head = data.get(at..at + 3).ok_or("a broken vocabulary")?;
            let len = usize::from(u16::from_le_bytes([head[0], head[1]]));
            special.push(head[2] != 0);
            bytes.push(
                data.get(at + 3..at + 3 + len)
                    .ok_or("a broken vocabulary")?
                    .to_vec(),
            );
            at += 3 + len;
        }
        Ok(Vocab { bytes, special })
    }

    /// The text of `ids`, special tokens left out.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = Vec::new();
        for &id in ids {
            let id = id as usize;
            if id < self.bytes.len() && !self.special[id] {
                out.extend_from_slice(&self.bytes[id]);
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }
}

/// The language and the words of what the model wrote: it writes
/// `language English<asr_text>The words…` (or `language None` for no
/// speech).
pub fn split_language(raw: &str) -> (String, String) {
    let raw = raw.trim();
    match raw.split_once("<asr_text>") {
        Some((head, words)) => {
            let language = head.trim().strip_prefix("language").unwrap_or(head).trim();
            (language.to_owned(), words.trim().to_owned())
        }
        None => (String::new(), raw.to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_byte_table_covers_every_byte() {
        let t = byte_of_char();
        assert_eq!(t.len(), 256);
        assert_eq!(t[&'A'], b'A');
        assert_eq!(t[&'\u{120}'], b' ');
        assert_eq!(t[&'\u{10a}'], b'\n');
    }

    #[test]
    fn decoding_spells_the_bytes_and_leaves_out_special_tokens() {
        let vocab = r#"{"Hello": 0, "Ġworld": 1, "!": 2}"#;
        let config = r#"{"added_tokens_decoder": {
            "3": {"content": "<|im_end|>", "special": true},
            "4": {"content": "<asr_text>", "special": false}}}"#;
        let v = Vocab::parse(vocab, config).unwrap();
        assert_eq!(v.decode(&[4, 0, 1, 2, 3]), "<asr_text>Hello world!");
    }

    #[test]
    fn the_language_comes_off_the_front() {
        assert_eq!(
            split_language("language English<asr_text>To survive."),
            ("English".into(), "To survive.".into())
        );
        assert_eq!(
            split_language("language None<asr_text>"),
            ("None".into(), String::new())
        );
    }

    #[test]
    fn the_prompt_wraps_the_audio() {
        let p = prompt(3);
        assert_eq!(p.len(), 9 + 3 + 6);
        assert_eq!(p.iter().filter(|&&i| i == AUDIO_PAD).count(), 3);
    }
}
