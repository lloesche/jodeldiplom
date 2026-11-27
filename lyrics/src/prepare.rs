//! The text to time against a recording, chosen by what a speech
//! recogniser heard in its voice ([`crate::asr`]): the text found when it
//! fits ([`crate::fit`]), else the best fitting of the song's other texts
//! when one does; with the repeats the recogniser heard and the text
//! leaves out put in when it fits ([`crate::repeats`]).

use crate::asr::Transcript;
use crate::fit::{fits, lyrics_fit};
use crate::words::words;
use crate::{repeats, Fit, Lyrics};

/// The least words heard for a fit to be judged (fewer: a voice too quiet
/// or too short to tell).
pub const HEARD_LEAST: usize = 20;

/// The song's other texts (LRCLIB's), asked for only when needed.
pub type Others<'a> = dyn Fn() -> Vec<Lyrics> + 'a;

/// Whether enough of `heard` (the recogniser's text) was heard to judge a
/// text by it, or to take it as the song's text: [`HEARD_LEAST`] words.
pub fn enough_heard(heard: &str) -> bool {
    words(heard).len() >= HEARD_LEAST
}

/// Of `lyrics` and `others`, the text to time against a voice in which the
/// recogniser `heard` this: `lyrics` when they fit ([`fits`]) or too
/// little was heard to judge ([`enough_heard`]), else the best fitting of
/// `others` when it fits, else `lyrics` still. And its fit, when judged.
pub fn fitting(lyrics: &Lyrics, heard: &str, others: &Others) -> (Lyrics, Option<Fit>) {
    if !enough_heard(heard) {
        return (lyrics.clone(), None);
    }
    let fit = lyrics_fit(lyrics, heard);
    if fits(fit) {
        return (lyrics.clone(), Some(Fit::new(fit, false)));
    }
    let best = others()
        .into_iter()
        .map(|o| {
            let f = lyrics_fit(&o, heard);
            (o, f)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1));
    match best {
        Some((other, f)) if fits(f) && f > fit => (other, Some(Fit::new(f, true))),
        _ => (lyrics.clone(), Some(Fit::new(fit, false))),
    }
}

/// What may be made of a text before it is timed ([`text`]).
pub enum Text<'a> {
    /// A text found by a source: the best fitting of the song's other
    /// texts (`others`, asked for only when needed) in its place when it
    /// does not fit, the repeats heard put in when it does.
    Found(&'a Others<'a>),
    /// A text edited by hand: timed as it is, never another in its place,
    /// nor repeats put in.
    Edited,
}

/// The text to time against a voice in which the recogniser `heard` this
/// (`None`: it was not heard), as `policy` lets it be made: `lyrics`, or
/// for a text found the best fitting of its others when it does not fit
/// ([`fitting`]), with the repeats heard and the text leaves out put in
/// when the text fits ([`repeats::with_repeats`]). And its fit, when
/// judged (never for a text edited).
pub fn text(lyrics: &Lyrics, heard: Option<&Transcript>, policy: &Text) -> (Lyrics, Option<Fit>) {
    let (Some(heard), Text::Found(others)) = (heard, policy) else {
        return (lyrics.clone(), None);
    };
    let (lyrics, fit) = fitting(lyrics, &heard.text, others);
    if fit.is_some_and(|f| f.fits()) {
        (repeats::with_repeats(&lyrics, &heard.parts), fit)
    } else {
        (lyrics, fit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Source;

    /// What the voice of the songs in these tests is heard to sing.
    const HEARD: &str = "You search through the years and you wipe away the tears. You find \
        a woman, not a girl, but a feeling like the beginning of the world.";

    #[test]
    fn a_text_that_fits_is_kept_and_one_that_does_not_is_swapped() {
        let text = |t: &str| Lyrics::parse_text(t, Source::Lrclib);
        let right = text(
            "You search through the years\nAnd you wipe away the tears\n\
             You find a woman, not a girl\nBut a feeling like the beginning of the world",
        );
        let wrong = text("Hello there\nGeneral Kenobi\nYou are a bold one");
        let asked = std::cell::Cell::new(0);
        let others = || {
            asked.set(asked.get() + 1);
            vec![wrong.clone(), right.clone()]
        };
        // Fits: kept, and the others never asked for.
        let (kept, fit) = fitting(&right, HEARD, &others);
        assert_eq!(kept, right);
        assert!(fit.unwrap().percent > 90 && !fit.unwrap().swapped);
        assert_eq!(asked.get(), 0);
        // Does not fit: the other that does.
        let (swapped, fit) = fitting(&wrong, HEARD, &others);
        assert_eq!(swapped, right);
        assert!(fit.unwrap().fits() && fit.unwrap().swapped);
        // None fits: kept as it is, its fit said.
        let (kept, fit) = fitting(&wrong, HEARD, &|| vec![wrong.clone()]);
        assert_eq!(kept, wrong);
        assert!(!fit.unwrap().fits() && !fit.unwrap().swapped);
        // Too little heard to judge: kept, not judged.
        assert!(!enough_heard("oh oh yeah"));
        assert_eq!(
            fitting(&wrong, "oh oh yeah", &others),
            (wrong.clone(), None)
        );
    }

    #[test]
    fn a_text_edited_or_not_heard_is_timed_as_it_is() {
        let text_of = |t: &str| Lyrics::parse_text(t, Source::Lrclib);
        let right = text_of(
            "You search through the years\nAnd you wipe away the tears\n\
             You find a woman, not a girl\nBut a feeling like the beginning of the world",
        );
        let wrong = text_of("Hello there\nGeneral Kenobi\nYou are a bold one");
        let heard = Transcript {
            language: String::new(),
            text: HEARD.into(),
            ids: Vec::new(),
            parts: Vec::new(),
        };
        let others = || vec![right.clone()];
        let found = Text::Found(&others);
        let (swapped, fit) = text(&wrong, Some(&heard), &found);
        assert_eq!(swapped, right);
        assert!(fit.unwrap().swapped);
        assert_eq!(
            text(&wrong, Some(&heard), &Text::Edited),
            (wrong.clone(), None)
        );
        assert_eq!(text(&wrong, None, &found), (wrong.clone(), None));
    }
}
