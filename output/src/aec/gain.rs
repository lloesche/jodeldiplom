//! The gain the filter in use's estimate takes: the loudness the speakers
//! play it at now.
//!
//! A speaker's limiter rides its loudness: as a passage 7 dB louder began,
//! a MacBook's speakers played it only 2 dB louder, and the filter learnt on
//! the quieter part overshot the echo by 8 dB for a second, so it seemed to
//! be sung over and was started over. Scaled by what the microphone holds
//! of it, an estimate comes nearer the echo. The gain only ever takes the
//! estimate down (a limiter only pulls), and is 1 while it predicts
//! nothing.
//!
//! What the estimate does not account for (a voice, what the filter leaves)
//! has nothing in common with it, yet over a few hundred samples moves the
//! gain by chance; through speakers that play the music as it is, cancelled
//! by 30 dB, that left a few dB more of the echo under a singer. So the
//! gain's step from 1 is shrunk by what chance makes of it (its variance:
//! the leftover's power over the estimate's and the samples the smoothing
//! spans, [`GAIN_SAMPLES`]): a limiter's pull of a few dB is taken whole,
//! chance's wobble not; and while someone sang lately it is smoothed ten
//! times slower.

use super::SMOOTH;

/// The smoothing while someone sang lately, per block (about 0.3 s).
const SLOW: f32 = 0.01;
/// The samples whose chance products move the gain as one, over the ~30 ms
/// it is smoothed over: music's samples are far from independent, a tenth
/// of them is about what counts.
const GAIN_SAMPLES: f32 = 250.0;

/// The gain ([`crate::aec::gain`]).
pub(super) struct Gain {
    /// The microphone's product with the estimate and the estimate's power,
    /// smoothed.
    cross: f32,
    power: f32,
    /// The gain the last block took.
    now: f32,
}

impl Gain {
    /// A gain that has seen nothing (1).
    pub(super) fn new() -> Gain {
        Gain {
            cross: 0.0,
            power: 0.0,
            now: 1.0,
        }
    }

    /// The gain the last block took.
    pub(super) fn now(&self) -> f32 {
        self.now
    }

    /// The gain for a block whose microphone's product with the estimate is
    /// `cross` and whose estimate's power is `power` (both a sample's mean),
    /// `sung` whether someone sang lately, `leftover` the power the filter
    /// in use leaves.
    pub(super) fn follow(&mut self, cross: f32, power: f32, sung: bool, leftover: f32) -> f32 {
        if !(cross.is_finite() && power.is_finite()) {
            *self = Gain::new();
            return 1.0;
        }
        let (by, samples) = if sung {
            (SLOW, GAIN_SAMPLES * SMOOTH / SLOW)
        } else {
            (SMOOTH, GAIN_SAMPLES)
        };
        self.cross += (cross - self.cross) * by;
        self.power += (power - self.power) * by;
        let tiny = 1e-12;
        let raw = (self.cross + tiny) / (self.power + tiny);
        let chance = leftover / (self.power * samples + tiny);
        let step = raw - 1.0;
        let kept = step * step / (step * step + chance + tiny);
        self.now = (1.0 + step * kept).clamp(0.0, 1.0);
        self.now
    }
}
