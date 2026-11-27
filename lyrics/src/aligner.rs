//! The CTC aligner put together: a speech model's emissions for the voice
//! ([`Acoustic`]), what a recogniser heard in it, and the alignment
//! ([`crate::ctc`]) held to a synced text's line times where they fit
//! ([`crate::scaffold`]).

use crate::asr::Part;
use crate::ctc::{voice_pauses, CtcSettings, Emissions, Hints};
use crate::tokens::{Vocab, WAV2VEC2_960H};
use crate::{spread, Lyrics, Maker, VERSION};

/// A speech model giving CTC emissions for the voice.
pub trait Acoustic {
    /// Its name and version.
    fn maker(&self) -> Maker;

    /// Its vocabulary.
    fn vocab(&self) -> Vocab {
        WAV2VEC2_960H
    }

    /// Its emissions for the voice at [`crate::VOCALS_RATE`]; `None` when
    /// it cannot run.
    fn emissions(&self, vocals_16k: &[f32]) -> Option<Emissions>;
}

/// A model shared between threads is a model.
impl<A: Acoustic + ?Sized> Acoustic for std::sync::Arc<A> {
    fn maker(&self) -> Maker {
        (**self).maker()
    }

    fn vocab(&self) -> Vocab {
        (**self).vocab()
    }

    fn emissions(&self, vocals_16k: &[f32]) -> Option<Emissions> {
        (**self).emissions(vocals_16k)
    }
}

/// The name and version a [`CtcAligner`] over a model made by `model`
/// records in its results: the model's, and this crate's timing
/// ([`VERSION`]). A timing kept by another is made again.
pub fn aligner_maker(model: &Maker) -> Maker {
    Maker::new(format!("ctc:{}@{}", model.name, model.version), VERSION)
}

/// The CTC aligner: a model's emissions, then [`crate::scaffold::align`].
/// Without emissions, or with no path, the times come from
/// [`spread::fallback`], but for the words set by hand
/// ([`crate::WordEdit::pin`]), which keep theirs.
pub struct CtcAligner<A> {
    /// The model.
    pub model: A,
    /// How the alignment goes.
    pub settings: CtcSettings,
}

impl<A: Acoustic> CtcAligner<A> {
    /// An aligner over `model` with the default settings.
    pub fn new(model: A) -> CtcAligner<A> {
        CtcAligner {
            model,
            settings: CtcSettings::default(),
        }
    }

    /// The name and version it records in its results ([`aligner_maker`]).
    pub fn maker(&self) -> Maker {
        aligner_maker(&self.model.maker())
    }

    /// `lyrics` with word times, from the voice at [`crate::VOCALS_RATE`]
    /// (mono), with what a recogniser heard in it (its transcript's `parts`
    /// and its `glimpses`, none when it was not heard) pinning the words to
    /// where it heard them ([`crate::anchors`], [`crate::glimpses`]), and
    /// the words set by hand held to their times, the others fitted around
    /// them ([`crate::WordEdit::pin`]). The result's [`crate::Timing::by`]
    /// names this aligner.
    pub fn align(
        &self,
        vocals_16k: &[f32],
        lyrics: &Lyrics,
        parts: &[Part],
        glimpses: &[Part],
    ) -> Lyrics {
        let aligned = self.model.emissions(vocals_16k).and_then(|e| {
            let hints = Hints {
                pauses: voice_pauses(vocals_16k, &e),
                windows: crate::anchors::windows(lyrics, parts),
                glimpses: glimpses.to_vec(),
            };
            crate::scaffold::align(&e, lyrics, &self.model.vocab(), &self.settings, &hints)
        });
        match aligned {
            Some(mut out) => {
                out.timing.by = Some(self.maker());
                out
            }
            None => {
                let mut out = spread::fallback(vocals_16k, lyrics);
                for w in out.lines.iter_mut().flat_map(|l| &mut l.words) {
                    w.keep_pin();
                }
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctc::tests::{emissions_of, lyrics, plant, FS};

    struct Planted(Emissions);

    impl Acoustic for Planted {
        fn maker(&self) -> Maker {
            Maker::new("planted", 7)
        }
        fn emissions(&self, _: &[f32]) -> Option<Emissions> {
            Some(self.0.clone())
        }
    }

    #[test]
    fn the_aligner_records_itself() {
        let mut on = vec![WAV2VEC2_960H.blank; 200];
        plant(&mut on, "hi", 20, 2);
        let aligner = CtcAligner::new(Planted(emissions_of(&on, 0.9)));
        let out = aligner.align(&[], &lyrics(&["hi"]), &[], &[]);
        assert_eq!(out.timing.by, Some(Maker::new("ctc:planted@7", VERSION)));
        assert_eq!(out.lines[0].words[0].start, Some(20.0 * FS));
    }

    /// A word set by hand where the voice sings the next one: it stays
    /// exactly where it was set, with its letters inside it, and the next
    /// word fits in after it; the one before stays where it is heard.
    /// Aligned again (a new version of the aligner), the same.
    #[test]
    fn a_word_set_by_hand_stays_and_its_neighbours_fit_around_it() {
        let mut on = vec![WAV2VEC2_960H.blank; 300];
        let one = plant(&mut on, "one", 20, 2);
        plant(&mut on, "two", 60, 2);
        plant(&mut on, "three", 150, 2);
        let aligner = CtcAligner::new(Planted(emissions_of(&on, 0.9)));
        let free = aligner.align(&[], &lyrics(&["one two three"]), &[], &[]);
        let at = |l: &Lyrics, k: usize| {
            let w = &l.lines[0].words[k];
            (w.start.unwrap(), w.end.unwrap())
        };
        assert_eq!(at(&free, 2).0, 150.0 * FS);
        // "two" moved by hand over where "three" is heard.
        let mut moved = free.clone();
        let two = &mut moved.lines[0].words[1];
        two.pin_to(140.0 * FS, 170.0 * FS);
        for _ in 0..2 {
            let out = aligner.align(&[], &moved, &[], &[]);
            assert_eq!(at(&out, 1), (140.0 * FS, 170.0 * FS));
            let two = &out.lines[0].words[1];
            assert!(two.edit.pin.is_some() && two.confidence == crate::GIVEN);
            assert_eq!(two.letters.len(), 3);
            assert!(two.letters.iter().all(|&t| (0.0..=0.6 + 1e-6).contains(&t)));
            assert_eq!(at(&out, 0), (20.0 * FS, one as f64 * FS));
            assert!(at(&out, 2).0 >= 170.0 * FS, "{:?}", at(&out, 2));
            moved = out;
        }
    }

    /// Without the model's emissions, a word set by hand keeps its times
    /// all the same.
    #[test]
    fn a_word_set_by_hand_keeps_its_times_without_the_model() {
        struct Deaf;
        impl Acoustic for Deaf {
            fn maker(&self) -> Maker {
                Maker::new("deaf", 1)
            }
            fn emissions(&self, _: &[f32]) -> Option<Emissions> {
                None
            }
        }
        let mut text = lyrics(&["one two three"]);
        let two = &mut text.lines[0].words[1];
        two.pin_to(3.0, 3.5);
        let out = CtcAligner::new(Deaf).align(&[0.0; 16_000], &text, &[], &[]);
        let two = &out.lines[0].words[1];
        assert_eq!((two.start, two.end), (Some(3.0), Some(3.5)));
    }
}
