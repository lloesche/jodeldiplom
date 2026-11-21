//! The words as a speech model's tokens.
//!
//! A CTC speech model such as `facebook/wav2vec2-base-960h` gives, for
//! every 20 ms of voice, how likely each of its letters is, or a blank.
//! Its letters are the 26 upper-case ones and `'`, with `|` between words.
//! A word is made into those letters by:
//!
//! - upper case;
//! - diacritics stripped (`ä` → `A`, `ç` → `C`), and a few letters spelt
//!   out (`ß` → `SS`, `æ` → `AE`, `þ` → `TH`);
//! - `'` kept inside a word (`DON'T`), curly apostrophes made straight,
//!   and leading or trailing ones dropped;
//! - a hyphen or dash inside a word made a word break (`|`);
//! - other punctuation dropped.
//!
//! A word with a letter of another script (Cyrillic, CJK, ...) or a digit
//! cannot be spelt in these letters: it is left out of the tokens and
//! timed between its neighbours instead.

use std::ops::Range;

use crate::Lyrics;

/// A model's tokens, by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vocab {
    /// Each token's text, at its index.
    pub tokens: &'static [&'static str],
    /// The index of the blank (CTC's "nothing new here").
    pub blank: u32,
    /// The index of the word break.
    pub separator: u32,
}

/// The vocabulary of `facebook/wav2vec2-base-960h` (its `vocab.json`):
/// `<pad>` is the blank.
pub const WAV2VEC2_960H: Vocab = Vocab {
    tokens: &[
        "<pad>", "<s>", "</s>", "<unk>", "|", "E", "T", "A", "O", "N", "I", "H", "S", "R", "D",
        "L", "U", "M", "W", "C", "F", "G", "Y", "P", "B", "V", "K", "'", "X", "J", "Q", "Z",
    ],
    blank: 0,
    separator: 4,
};

impl Vocab {
    /// The index of the token for `c` (an upper-case letter, `'` or `|`).
    pub fn id(&self, c: char) -> Option<u32> {
        let mut buffer = [0u8; 4];
        let text: &str = c.encode_utf8(&mut buffer);
        self.tokens
            .iter()
            .position(|t| *t == text)
            .map(|i| i as u32)
    }

    /// How many tokens there are.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// `c` without its diacritics, in upper case, when it is a Latin letter:
/// one letter, or two or three for a ligature or `ß`.
pub(crate) fn latin(c: char) -> Option<&'static str> {
    if c.is_ascii_alphabetic() {
        const LETTERS: [&str; 26] = [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
            "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        ];
        return Some(LETTERS[(c.to_ascii_uppercase() as u8 - b'A') as usize]);
    }
    let code = c as u32;
    // Latin-1: the lower-case letters sit 0x20 above the upper-case ones.
    let latin1 = |code: u32| -> Option<&'static str> {
        Some(match code {
            0xC0..=0xC5 => "A",
            0xC6 => "AE",
            0xC7 => "C",
            0xC8..=0xCB => "E",
            0xCC..=0xCF => "I",
            0xD0 => "D",
            0xD1 => "N",
            0xD2..=0xD6 | 0xD8 => "O",
            0xD9..=0xDC => "U",
            0xDD => "Y",
            0xDE => "TH",
            _ => return None,
        })
    };
    match code {
        0xDF => Some("SS"),
        0xFF => Some("Y"),
        0xC0..=0xDE => latin1(code),
        0xE0..=0xFE => latin1(code - 0x20),
        // Latin Extended-A: upper and lower case in pairs.
        0x100..=0x105 => Some("A"),
        0x106..=0x10D => Some("C"),
        0x10E..=0x111 => Some("D"),
        0x112..=0x11B => Some("E"),
        0x11C..=0x123 => Some("G"),
        0x124..=0x127 => Some("H"),
        0x128..=0x131 => Some("I"),
        0x132..=0x133 => Some("IJ"),
        0x134..=0x135 => Some("J"),
        0x136..=0x138 => Some("K"),
        0x139..=0x142 => Some("L"),
        0x143..=0x14B => Some("N"),
        0x14C..=0x151 => Some("O"),
        0x152..=0x153 => Some("OE"),
        0x154..=0x159 => Some("R"),
        0x15A..=0x161 | 0x17F => Some("S"),
        0x162..=0x167 => Some("T"),
        0x168..=0x173 => Some("U"),
        0x174..=0x175 => Some("W"),
        0x176..=0x178 => Some("Y"),
        0x179..=0x17E => Some("Z"),
        _ => None,
    }
}

/// Whether `c` is an apostrophe of some kind.
fn is_apostrophe(c: char) -> bool {
    matches!(c, '\'' | '’' | '‘' | 'ʼ' | '`' | '´')
}

/// `word` in the model's letters, with `|` where a hyphen broke it; `None`
/// when it has a digit, a letter of another script, or no letter at all.
pub fn normalize(word: &str) -> Option<String> {
    let mut out = String::new();
    for c in word.chars() {
        if let Some(letters) = latin(c) {
            out.push_str(letters);
        } else if is_apostrophe(c) {
            out.push('\'');
        } else if matches!(c, '-' | '‐' | '‑' | '–' | '—') {
            out.push('|');
        } else if ('\u{300}'..='\u{36f}').contains(&c) {
            // A combining diacritic, from decomposed text.
        } else if c.is_alphanumeric() {
            return None;
        }
    }
    // Apostrophes and breaks only between letters.
    let mut cleaned = String::with_capacity(out.len());
    let parts = out
        .split('|')
        .map(|part| part.trim_matches('\''))
        .filter(|part| !part.is_empty());
    for part in parts {
        if !cleaned.is_empty() {
            cleaned.push('|');
        }
        cleaned.push_str(part);
    }
    (!cleaned.is_empty()).then_some(cleaned)
}

/// Where one word of the lyrics sits in the token sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenWord {
    /// Its line.
    pub line: usize,
    /// Its place in the line.
    pub word: usize,
    /// Its tokens, or `None` when it cannot be spelt in the model's
    /// letters (then it is timed between its neighbours).
    pub tokens: Option<Range<usize>>,
    /// The tokens the alignment may leave out with it, when it is a
    /// backing word: its own and one word break beside it.
    pub skip: Option<Range<usize>>,
}

/// The lyrics as one token sequence.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tokens {
    /// The token indices: the words' letters, joined by the word break.
    pub ids: Vec<u32>,
    /// Every word of the lyrics, in order, and where its tokens are.
    pub words: Vec<TokenWord>,
}

/// The tokens of `lyrics` in `vocab`.
pub fn tokenize(lyrics: &Lyrics, vocab: &Vocab) -> Tokens {
    let mut tokens = Tokens::default();
    // Whether each spelt word is a backing one, with its index in `words`.
    let mut spelt = Vec::new();
    for (l, line) in lyrics.lines.iter().enumerate() {
        for (w, word) in line.words.iter().enumerate() {
            let ids: Option<Vec<u32>> =
                normalize(&word.text).and_then(|text| text.chars().map(|c| vocab.id(c)).collect());
            let range = ids.filter(|ids| !ids.is_empty()).map(|ids| {
                if !tokens.ids.is_empty() {
                    tokens.ids.push(vocab.separator);
                }
                let start = tokens.ids.len();
                tokens.ids.extend(ids);
                start..tokens.ids.len()
            });
            if range.is_some() {
                spelt.push((tokens.words.len(), word.backing));
            }
            tokens.words.push(TokenWord {
                line: l,
                word: w,
                tokens: range,
                skip: None,
            });
        }
    }
    // A backing word leaves out the word break after it, so the words
    // either side still meet at one; at the end of the song (no sung word
    // after it), the break before it.
    for (k, &(i, backing)) in spelt.iter().enumerate() {
        if !backing {
            continue;
        }
        let Some(range) = tokens.words[i].tokens.clone() else {
            continue;
        };
        let sung_after = spelt[k + 1..].iter().any(|&(_, b)| !b);
        let skip = match spelt.get(k + 1) {
            Some(&(next, _)) if sung_after => {
                let next = tokens.words[next]
                    .tokens
                    .as_ref()
                    .map_or(range.end, |r| r.start);
                range.start..next
            }
            _ if range.start > 0 => range.start - 1..range.end,
            _ => range,
        };
        tokens.words[i].skip = Some(skip);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Line, Source, Timing};

    #[test]
    fn the_vocabulary_is_wav2vec2s() {
        let v = WAV2VEC2_960H;
        assert_eq!(v.len(), 32);
        assert_eq!(v.id('|'), Some(4));
        assert_eq!(v.id('E'), Some(5));
        assert_eq!(v.id('\''), Some(27));
        assert_eq!(v.id('Z'), Some(31));
        assert_eq!(v.id('e'), None);
        for c in 'A'..='Z' {
            assert!(v.id(c).is_some(), "{c}");
        }
    }

    #[test]
    fn words_are_upper_case_without_diacritics() {
        assert_eq!(normalize("Hello,").as_deref(), Some("HELLO"));
        assert_eq!(normalize("don’t").as_deref(), Some("DON'T"));
        assert_eq!(normalize("'cause").as_deref(), Some("CAUSE"));
        assert_eq!(normalize("rockin'").as_deref(), Some("ROCKIN"));
        assert_eq!(normalize("Mädchen").as_deref(), Some("MADCHEN"));
        assert_eq!(normalize("Straße").as_deref(), Some("STRASSE"));
        assert_eq!(normalize("Ærø").as_deref(), Some("AERO"));
        assert_eq!(normalize("Łódź").as_deref(), Some("LODZ"));
        assert_eq!(normalize("e\u{301}te\u{301}").as_deref(), Some("ETE"));
        assert_eq!(normalize("well-known").as_deref(), Some("WELL|KNOWN"));
        assert_eq!(normalize("(yeah)").as_deref(), Some("YEAH"));
        assert_eq!(normalize("—"), None);
        assert_eq!(normalize("99"), None);
        assert_eq!(normalize("愛してる"), None);
        assert_eq!(normalize("любовь"), None);
    }

    #[test]
    fn lyrics_are_joined_by_word_breaks_and_unspellable_words_flagged() {
        let lyrics = Lyrics {
            lines: vec![Line::from_text("Oh, 99 la"), Line::from_text("ah (ooh) ay")],
            source: Source::Unknown,
            timing: Timing::default(),
        };
        let t = tokenize(&lyrics, &WAV2VEC2_960H);
        let text: String = t
            .ids
            .iter()
            .map(|&i| WAV2VEC2_960H.tokens[i as usize])
            .collect();
        assert_eq!(text, "OH|LA|AH|OOH|AY");
        let ranges: Vec<Option<Range<usize>>> = t.words.iter().map(|w| w.tokens.clone()).collect();
        assert_eq!(
            ranges,
            [
                Some(0..2),
                None,
                Some(3..5),
                Some(6..8),
                Some(9..12),
                Some(13..15)
            ]
        );
        assert_eq!((t.words[1].line, t.words[1].word), (0, 1));
        assert_eq!((t.words[4].line, t.words[4].word), (1, 1));
        // The backing word leaves out the break after it.
        assert_eq!(t.words[4].skip, Some(9..13));
        assert!(t
            .words
            .iter()
            .enumerate()
            .all(|(i, w)| (i == 4) == w.skip.is_some()));
    }

    #[test]
    fn backing_words_at_the_end_leave_out_the_break_before() {
        let lyrics = Lyrics {
            lines: vec![Line::from_text("go (go go)")],
            source: Source::Unknown,
            timing: Timing::default(),
        };
        let t = tokenize(&lyrics, &WAV2VEC2_960H);
        // GO|GO|GO
        assert_eq!(t.words[1].skip, Some(2..5));
        assert_eq!(t.words[2].skip, Some(5..8));
    }
}
