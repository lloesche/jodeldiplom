//! The canceller and the suppressor over whole signals, offline: for tests,
//! replays of recordings and simulations.

use super::{EchoCanceller, EchoStatus, Suppressor, BLOCK};

/// A canceller and the suppressor after it, run over whole signals
/// ([`Whole::run`]); what they learnt carries over from one run to the
/// next, as in one session.
pub struct Whole {
    canceller: EchoCanceller,
    suppressor: Suppressor,
}

/// What [`Whole::run`] gives: the canceller's voice and the suppressed one,
/// both lined up with the microphone (the suppressor's block of delay taken
/// back), as long as it in whole blocks.
pub struct Cleaned {
    /// The canceller's voice (the microphone less the echo it predicts).
    pub cancelled: Vec<f32>,
    /// That voice with its leftover suppressed (as the game hears it).
    pub suppressed: Vec<f32>,
}

impl Whole {
    /// Both at `rate`.
    pub fn new(rate: u32) -> Whole {
        Whole {
            canceller: EchoCanceller::new(rate),
            suppressor: Suppressor::new(rate),
        }
    }

    /// The canceller, as it is now.
    pub fn canceller(&self) -> &EchoCanceller {
        &self.canceller
    }

    /// Runs over `mic` and what played, `left` and `right` (the same for
    /// one speaker), all as long; `each` is told after every block where it
    /// ended (in samples) and how the canceller is doing.
    pub fn run(
        &mut self,
        mic: &[f32],
        left: &[f32],
        right: &[f32],
        mut each: impl FnMut(usize, &EchoStatus),
    ) -> Cleaned {
        let n = mic.len().min(left.len()).min(right.len()) / BLOCK * BLOCK;
        let mut cleaned = Cleaned {
            cancelled: vec![0.0; n],
            suppressed: vec![0.0; n],
        };
        let mut late = [0.0f32; BLOCK];
        // A run is a stream of its own: nothing of the last one's carries.
        self.suppressor.restart();
        for start in (0..n).step_by(BLOCK) {
            let r = start..start + BLOCK;
            let block = &mut cleaned.cancelled[r.clone()];
            let echo = self
                .canceller
                .process(&mic[r.clone()], &left[r.clone()], &right[r], block);
            self.suppressor.process(block, &echo, &mut late);
            // What comes out is the block before's.
            if start >= BLOCK {
                cleaned.suppressed[start - BLOCK..start].copy_from_slice(&late);
            }
            each(start + BLOCK, &self.canceller.status());
        }
        cleaned
    }
}
