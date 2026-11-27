//! Whether a text fits a recording: its words against what a speech
//! recogniser heard in the voice ([`crate::asr`]). The recogniser gets
//! about a third of a song's words wrong, but a text that fits still has
//! most of its words heard, in order. A text of another song, another
//! version or another language has few.
//!
//! On a bake-off of 100 songs a fit under [`FIT_LEAST_PERCENT`] picked out
//! the texts that did not fit (wrong song, wrong version, another
//! language's words); texts that fit had a median of about three quarters.

use crate::words::{common, words, Compared};
use crate::Lyrics;

/// A text fits when at least this share of its words were heard, in whole
/// percent.
pub const FIT_LEAST_PERCENT: u8 = 35;

/// `share` (0 to 1) in whole percent, rounded down.
pub fn percent_of(share: f32) -> u8 {
    (share.clamp(0.0, 1.0) * 100.0).floor() as u8
}

/// Whether a text of which `share` (0 to 1) of the words were heard fits
/// ([`FIT_LEAST_PERCENT`]), judged on its whole percent ([`percent_of`]),
/// as a kept [`crate::Fit`] is.
pub fn fits(share: f32) -> bool {
    percent_of(share) >= FIT_LEAST_PERCENT
}

/// The share of `text`'s words that were `heard`, in order (the most in
/// [`common`], not necessarily together), 0 to 1; 0 for no words.
pub fn fit(text: &[String], heard: &[String]) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    common(text, heard).len() as f32 / text.len() as f32
}

/// How well `lyrics` fit what was `heard` (the recogniser's text).
pub fn lyrics_fit(lyrics: &Lyrics, heard: &str) -> f32 {
    fit(&Compared::of(lyrics).words, &words(heard))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Source;

    #[test]
    fn a_text_fits_by_the_share_of_its_words_heard_in_order() {
        let text = words("look at the hate we're breeding look at the fear");
        let heard = words("look at the heat we're breathing look at fear");
        let f = fit(&text, &heard);
        assert!((f - 0.7).abs() < 1e-6, "{f}");
        // The same words, out of order, count once each in order.
        let shuffled = words("fear the at look breeding we're hate the at look");
        assert!(!fits(fit(&text, &shuffled)));
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
        assert!(!fits(lyrics_fit(&lyrics, other)));
    }
}
