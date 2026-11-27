//! Words as they are compared: a text's words loosely (case, curly
//! apostrophes and punctuation aside), the lyrics' words each with the
//! word and line they are of ([`Compared`]), the words two lists have in
//! common, in order ([`common`]), and a recogniser's part that fell into
//! a loop ([`looped`]). The fit ([`crate::fit`]), the pins
//! ([`crate::anchors`]), the repeats ([`crate::repeats`]), the scaffold
//! ([`crate::scaffold`]) and the glimpses ([`crate::glimpses`]) all
//! compare words so.

use std::collections::HashSet;

use crate::asr::Part;
use crate::Lyrics;

/// Whether `c` is written without spaces between words (Chinese,
/// Japanese, Korean): such text is compared a character at a time.
fn unspaced(c: char) -> bool {
    matches!(c,
        '\u{3040}'..='\u{30ff}' // Hiragana, Katakana
        | '\u{3400}'..='\u{4dbf}' // CJK extension A
        | '\u{4e00}'..='\u{9fff}' // CJK ideographs
        | '\u{ac00}'..='\u{d7af}' // Hangul syllables
        | '\u{f900}'..='\u{faff}') // CJK compatibility ideographs
}

/// The words of `text` for comparing: lower case, curly apostrophes made
/// straight, letters, digits and apostrophes only (an apostrophe at a
/// word's ends dropped), each unspaced character a word of its own.
pub fn words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut Vec<String>| {
        let w = word.trim_matches('\'');
        if !w.is_empty() {
            out.push(w.to_owned());
        }
        word.clear();
    };
    for c in text.chars() {
        let c = if c == '\u{2019}' || c == '\u{2018}' {
            '\''
        } else {
            c
        };
        if unspaced(c) {
            flush(&mut word, &mut out);
            out.push(c.to_string());
        } else if c.is_alphanumeric() || c == '\'' {
            word.extend(c.to_lowercase());
        } else {
            flush(&mut word, &mut out);
        }
    }
    flush(&mut word, &mut out);
    out
}

/// The words of lyrics as compared ([`words`]), line after line, each
/// with the lyrics' word it is of (a word may compare as two:
/// "five-thirty", or as none: "...") and that word's line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Compared {
    /// The compared words, in order.
    pub words: Vec<String>,
    /// For each, the lyrics' word it is of, counted over all lines.
    pub word: Vec<usize>,
    /// For each, its line.
    pub line: Vec<usize>,
    /// For each line, and one past the last, the first of its compared
    /// words.
    starts: Vec<usize>,
}

impl Compared {
    /// The words of `lyrics` as compared.
    pub fn of(lyrics: &Lyrics) -> Compared {
        let mut out = Compared::default();
        let mut flat = 0;
        for (l, line) in lyrics.lines.iter().enumerate() {
            out.starts.push(out.words.len());
            for w in &line.words {
                for c in words(&w.text) {
                    out.words.push(c);
                    out.word.push(flat);
                    out.line.push(l);
                }
                flat += 1;
            }
        }
        out.starts.push(out.words.len());
        out
    }

    /// How many lines the lyrics have.
    pub fn lines(&self) -> usize {
        self.starts.len() - 1
    }

    /// The first compared word of `line` (of the lines after it when it
    /// has none; all the words' count for the line after the last).
    pub fn first_of(&self, line: usize) -> usize {
        self.starts[line]
    }

    /// The compared words of `line`.
    pub fn of_line(&self, line: usize) -> &[String] {
        &self.words[self.starts[line]..self.starts[line + 1]]
    }
}

/// The pairs `(i, j)` of a longest common subsequence of `a` and `b`: the
/// most of their words in common, in order, not necessarily together.
pub fn common<T: PartialEq>(a: &[T], b: &[T]) -> Vec<(usize, usize)> {
    let (n, m) = (a.len(), b.len());
    let mut t = vec![0u32; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            t[at(i, j)] = if a[i] == b[j] {
                t[at(i + 1, j + 1)] + 1
            } else {
                t[at(i + 1, j)].max(t[at(i, j + 1)])
            };
        }
    }
    let (mut i, mut j, mut out) = (0, 0, Vec::new());
    while i < n && j < m {
        if a[i] == b[j] {
            out.push((i, j));
            i += 1;
            j += 1;
        } else if t[at(i + 1, j)] >= t[at(i, j + 1)] {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// Whether the recogniser fell into a loop in `part`: more words than are
/// sung in its time (over 4 a second), most of them one phrase over again
/// (under a third of its runs of three words different).
pub fn looped(part: &Part) -> bool {
    let w = words(&part.text);
    let seconds = (part.end - part.start).max(1.0);
    if (w.len() as f64) < 4.0 * seconds || w.len() < 3 {
        return false;
    }
    let threes: HashSet<&[String]> = w.windows(3).collect();
    (threes.len() as f64) < (w.len() - 2) as f64 / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Line, Source, Timing};

    fn part(start: f64, end: f64, text: &str) -> Part {
        Part {
            start,
            end,
            text: text.to_owned(),
        }
    }

    #[test]
    fn words_are_compared_loosely() {
        assert_eq!(
            words("Don’t walk AWAY, 'cause I-love you!"),
            ["don't", "walk", "away", "cause", "i", "love", "you"]
        );
        assert_eq!(
            words("どうして黙って"),
            ["ど", "う", "し", "て", "黙", "っ", "て"]
        );
        assert_eq!(words("Ich hör' dich"), ["ich", "hör", "dich"]);
    }

    #[test]
    fn the_lyrics_words_know_their_word_and_line() {
        let lyrics = Lyrics {
            lines: ["Five-thirty, go", "...", "", "Home"]
                .iter()
                .map(|t| Line::from_text(t))
                .collect(),
            source: Source::Unknown,
            timing: Timing::default(),
        };
        let c = Compared::of(&lyrics);
        assert_eq!(c.words, ["five", "thirty", "go", "home"]);
        assert_eq!(c.word, [0, 0, 1, 3]);
        assert_eq!(c.line, [0, 0, 0, 3]);
        assert_eq!(c.lines(), 4);
        assert_eq!(
            (0..=4).map(|l| c.first_of(l)).collect::<Vec<_>>(),
            [0, 3, 3, 3, 4]
        );
        assert_eq!(c.of_line(0), ["five", "thirty", "go"]);
        assert!(c.of_line(1).is_empty());
    }

    #[test]
    fn the_words_in_common_are_found_in_order() {
        let a = words("look at the hate we're breeding");
        let b = words("look at the heat we're breathing");
        assert_eq!(common(&a, &b), [(0, 0), (1, 1), (2, 2), (4, 4)]);
        assert!(common(&a, &[]).is_empty());
    }

    #[test]
    fn a_loop_is_not_heard() {
        let looping = part(0.0, 10.0, &"cry me a river ".repeat(30));
        assert!(looped(&looping));
        // A chorus sung four times in a part is no loop; nor are fast
        // words that differ.
        assert!(!looped(&part(0.0, 26.0, &"cry me a river ".repeat(4))));
        let fast: String = (0..60).map(|i| format!("w{i} ")).collect();
        assert!(!looped(&part(0.0, 10.0, &fast)));
    }
}
