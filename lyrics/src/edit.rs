//! What was set on a word by hand ([`WordEdit`]), and a song's edits: its
//! whole text as edited, each word with only what was set on it
//! ([`Lyrics::edits`]), put back on a timing of that text
//! ([`Lyrics::with_edits`]), and what names such a timing
//! ([`Lyrics::edits_stamp`]).

use std::fmt::Write as _;

use crate::{Line, Lyrics, Timing, Word};

/// What was set on a word by hand. Its default is nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WordEdit {
    /// Its start and end set by hand, in seconds: the aligner keeps them,
    /// and fits the other words around it ([`crate::ctc`]). The word's own
    /// times are these ([`Word::keep_pin`]).
    pub pin: Option<(f64, f64)>,
    /// It is marked as not to be sung (a backing vocal, an ad-lib the
    /// singer leaves to the record): shown, but given no notes, so the
    /// game neither asks for it nor scores it.
    pub not_sung: bool,
    /// The voice stem's level over it, in dB against the voice as
    /// recorded; `None`: as the karaoke's voice level has it.
    pub voice_db: Option<f32>,
}

impl WordEdit {
    /// Whether anything was set.
    pub fn is_set(&self) -> bool {
        *self != WordEdit::default()
    }
}

impl Lyrics {
    /// What is kept of these lyrics (a song's as shown, edited): every line
    /// and word, each word with only what was set on it by hand (its times
    /// when it was moved, [`Word::keep_pin`]).
    pub fn edits(&self) -> Lyrics {
        let mut out = Lyrics {
            lines: Vec::with_capacity(self.lines.len()),
            source: self.source.clone(),
            timing: Timing::default(),
        };
        for line in &self.lines {
            let words = line
                .words
                .iter()
                .map(|w| {
                    let mut kept = Word::new(w.text.clone());
                    kept.edit = w.edit;
                    kept.keep_pin();
                    kept
                })
                .collect();
            let mut kept = Line {
                words,
                is_break: line.is_break,
                ..Line::default()
            };
            kept.mark_backing();
            out.lines.push(kept);
        }
        out
    }

    /// These lyrics (a timing of the text of `edits`) with what was set by
    /// hand on each word of `edits` put back on them: its marks, its times.
    /// `None` when the two are not the same words.
    pub fn with_edits(&self, edits: &Lyrics) -> Option<Lyrics> {
        let same = self.lines.len() == edits.lines.len()
            && self.lines.iter().zip(&edits.lines).all(|(a, b)| {
                a.words.len() == b.words.len()
                    && a.words.iter().zip(&b.words).all(|(x, y)| x.text == y.text)
            });
        if !same {
            return None;
        }
        let mut out = self.clone();
        for (line, by) in out.lines.iter_mut().zip(&edits.lines) {
            for (w, e) in line.words.iter_mut().zip(&by.words) {
                w.edit = e.edit;
                w.keep_pin();
            }
        }
        Some(out)
    }

    /// What names a timing made with these edits: a hash of their words
    /// and the times set by hand (not the marks, which do not change the
    /// timing).
    pub fn edits_stamp(&self) -> u64 {
        let mut text = String::new();
        for line in &self.lines {
            text.push(if line.is_break { 'b' } else { 'l' });
            for w in &line.words {
                let pin = w.edit.pin.map(|(a, b)| (a.to_bits(), b.to_bits()));
                let _ = write!(text, " {}|{pin:?}", w.text);
            }
            text.push('\n');
        }
        jodel_dsp::fnv1a(text.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Source, GIVEN};

    /// Edits as the editor leaves them: a word moved, one not to be sung,
    /// one with the voice set over it, a break.
    fn sample() -> Lyrics {
        let mut l = Lyrics::parse_text("Hold me (close)\n♪\nNever let go\n", Source::Lrclib);
        l.lines[0].words[1].pin_to(12.34, 12.9);
        l.lines[0].words[2].edit.not_sung = true;
        let w = &mut l.lines[2].words[0];
        w.pin_to(20.0, 20.5);
        (w.edit.not_sung, w.edit.voice_db) = (true, Some(-3.0));
        l.edits()
    }

    #[test]
    fn what_is_kept_of_edits_is_what_they_are() {
        let edits = sample();
        assert!(edits.lines[1].is_break);
        assert!(edits.lines[0].words[2].backing);
        let moved = &edits.lines[0].words[1];
        assert_eq!((moved.start, moved.end), (Some(12.34), Some(12.9)));
        assert_eq!(moved.confidence, GIVEN);
        assert_eq!(edits.edits(), edits);
    }

    #[test]
    fn a_timing_is_named_by_the_words_and_their_times_not_the_marks() {
        let edits = sample();
        let mut marked_only = edits.clone();
        marked_only.lines[2].words[1].edit.not_sung = true;
        assert_eq!(edits.edits_stamp(), marked_only.edits_stamp());
        let mut moved = edits.clone();
        moved.lines[0].words[1].pin_to(12.34, 13.0);
        assert_ne!(edits.edits_stamp(), moved.edits_stamp());
        let mut reworded = edits.clone();
        reworded.lines[2].words[2].text = "go!".into();
        assert_ne!(edits.edits_stamp(), reworded.edits_stamp());
    }

    #[test]
    fn a_kept_timing_gets_the_marks_back_and_only_for_the_same_words() {
        let edits = sample();
        let mut timed = edits.clone();
        for (k, w) in timed
            .lines
            .iter_mut()
            .flat_map(|l| &mut l.words)
            .enumerate()
        {
            (w.start, w.end) = (Some(k as f64), Some(k as f64 + 0.5));
            w.edit = WordEdit::default();
        }
        let back = timed.with_edits(&edits).unwrap();
        assert_eq!(back.edits(), edits);
        assert_eq!(back.lines[0].words[0].start, Some(0.0));
        assert_eq!(back.lines[0].words[1].start, Some(12.34));
        timed.lines[0].words.pop();
        assert_eq!(timed.with_edits(&edits), None);
    }
}
