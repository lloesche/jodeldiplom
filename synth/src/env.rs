//! An [`Adsr`] running: where a voice's amplitude envelope is, a block at a
//! time.

use crate::patch::Adsr;

/// How far an exponential segment has fallen at its end: 60 dB.
const FLOOR: f32 = 0.001;

/// Where an [`Envelope`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Stage {
    Attack,
    Decay,
    Sustain,
    Release,
    Done,
}

/// An [`Adsr`] running.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Envelope {
    stage: Stage,
    /// Seconds into the stage.
    t: f32,
    /// The level the stage started from.
    from: f32,
    level: f32,
}

impl Envelope {
    /// One starting its attack from `level`.
    pub(crate) fn start(level: f32) -> Envelope {
        Envelope {
            stage: Stage::Attack,
            t: 0.0,
            from: level,
            level,
        }
    }

    /// Its level now.
    pub(crate) fn level(&self) -> f32 {
        self.level
    }

    /// Whether the note is still held.
    pub(crate) fn held(&self) -> bool {
        self.stage < Stage::Release
    }

    /// Whether it has ended.
    pub(crate) fn done(&self) -> bool {
        self.stage == Stage::Done
    }

    /// Lets the note go: the release from where it is.
    pub(crate) fn release(&mut self) {
        if self.held() {
            self.stage = Stage::Release;
            self.t = 0.0;
            self.from = self.level;
        }
    }

    /// `dt` seconds on; its level then.
    pub(crate) fn advance(&mut self, a: &Adsr, mut dt: f32) -> f32 {
        loop {
            let (length, end, next) = match self.stage {
                Stage::Attack => (a.attack, 1.0, Stage::Decay),
                Stage::Decay => (a.decay, a.sustain, Stage::Sustain),
                Stage::Release => (a.release, 0.0, Stage::Done),
                Stage::Sustain | Stage::Done => break,
            };
            if self.t + dt < length {
                self.t += dt;
                break;
            }
            dt -= (length - self.t).max(0.0);
            self.stage = if next == Stage::Sustain && a.sustain <= 0.0 {
                Stage::Done
            } else {
                next
            };
            self.t = 0.0;
            self.from = end;
            self.level = end;
        }
        self.level = match self.stage {
            Stage::Attack => self.from + (1.0 - self.from) * self.t / a.attack,
            Stage::Decay => a.sustain + (1.0 - a.sustain) * FLOOR.powf(self.t / a.decay),
            Stage::Sustain => a.sustain,
            Stage::Release => self.from * FLOOR.powf(self.t / a.release),
            Stage::Done => 0.0,
        };
        self.level
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_envelope_rises_falls_holds_and_ends() {
        let a = Adsr {
            attack: 0.01,
            decay: 0.1,
            sustain: 0.5,
            release: 0.2,
        };
        let mut e = Envelope::start(0.0);
        assert!((e.advance(&a, 0.005) - 0.5).abs() < 1e-6);
        assert!((e.advance(&a, 0.005) - 1.0).abs() < 1e-3);
        let mid = e.advance(&a, 0.05);
        assert!(mid > 0.5 && mid < 1.0, "{mid}");
        assert_eq!(e.advance(&a, 1.0), 0.5);
        assert!(e.held());
        e.release();
        let half = e.advance(&a, 0.1);
        assert!((half - 0.5 * FLOOR.sqrt()).abs() < 1e-4, "{half}");
        assert_eq!(e.advance(&a, 0.1), 0.0);
        assert!(e.done());
    }

    #[test]
    fn a_percussive_envelope_ends_by_itself() {
        let a = Adsr {
            attack: 0.0,
            decay: 0.05,
            sustain: 0.0,
            release: 0.1,
        };
        let mut e = Envelope::start(0.0);
        assert!(e.advance(&a, 0.001) > 0.8);
        e.advance(&a, 0.06);
        assert!(e.done());
    }
}
