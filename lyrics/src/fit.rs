//! Whether a text fits a recording: its words against what a speech
//! recogniser heard in the voice ([`crate::asr`]). The recogniser gets
//! about a third of a song's words wrong, but a text that fits still has
//! most of its words heard, in order. A text of another song, another
//! version or another language has few.
//!
//! On a bake-off of 100 songs a fit under [`FIT_LEAST`] picked out the
//! texts that did not fit (wrong song, wrong version, another language's
//! words); texts that fit had a median of about three quarters.

use crate::Lyrics;

/// A text fits when at least this share of its words were heard.
pub const FIT_LEAST: f32 = 0.35;

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

/// The words of `lyrics`, in order ([`words`]).
pub fn lyrics_words(lyrics: &Lyrics) -> Vec<String> {
    lyrics
        .lines
        .iter()
        .flat_map(|l| l.words.iter())
        .flat_map(|w| words(&w.text))
        .collect()
}

/// The share of `text`'s words that were `heard`, in order (the longest
/// run common to both, not necessarily together), 0 to 1; 0 for no words.
pub fn fit(text: &[String], heard: &[String]) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    // The longest common subsequence, a row at a time.
    let mut prev = vec![0u32; heard.len() + 1];
    let mut row = vec![0u32; heard.len() + 1];
    for t in text {
        for (j, h) in heard.iter().enumerate() {
            row[j + 1] = if t == h {
                prev[j] + 1
            } else {
                row[j].max(prev[j + 1])
            };
        }
        std::mem::swap(&mut prev, &mut row);
    }
    prev[heard.len()] as f32 / text.len() as f32
}

/// How well `lyrics` fit what was `heard` (the recogniser's text).
pub fn lyrics_fit(lyrics: &Lyrics, heard: &str) -> f32 {
    fit(&lyrics_words(lyrics), &words(heard))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Source;

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
    fn a_text_fits_by_the_share_of_its_words_heard_in_order() {
        let text = words("look at the hate we're breeding look at the fear");
        let heard = words("look at the heat we're breathing look at fear");
        let f = fit(&text, &heard);
        assert!((f - 0.7).abs() < 1e-6, "{f}");
        // The same words, out of order, count once each in order.
        let shuffled = words("fear the at look breeding we're hate the at look");
        assert!(fit(&text, &shuffled) < 0.35);
        assert_eq!(fit(&[], &heard), 0.0);
    }

    #[test]
    fn another_songs_text_does_not_fit() {
        let lyrics = Lyrics::parse_text(
            "You search through the years\nAnd you wipe away the tears",
            Source::Lrclib,
        );
        let heard = "You search through the years, and you wipe away the tears.";
        assert!(lyrics_fit(&lyrics, heard) > 0.9);
        let other = "Miracles will happen as we speak, but we're never gonna survive";
        assert!(lyrics_fit(&lyrics, other) < FIT_LEAST);
    }
}
