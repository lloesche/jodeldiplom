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
/// [`spread::fallback`].
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
    /// where it heard them ([`crate::anchors`], [`crate::glimpses`]). The
    /// result's [`crate::Timing::by`] names this aligner.
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
            Some((mut out, _)) => {
                out.timing.by = Some(self.maker());
                out
            }
            None => spread::fallback(vocals_16k, lyrics),
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
}
