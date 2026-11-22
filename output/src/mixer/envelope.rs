//! A gain over a chunk of frames, as the mixer thread works it out: one for
//! all of them, or one per frame. It keeps its buffer from chunk to chunk,
//! so it grows to the largest chunk once and never allocates after.

use super::mix::Gains;
use super::ramp::{Gain, When};

/// A gain over the frames of a chunk.
#[derive(Clone, Debug, Default)]
pub(crate) struct Envelope {
    /// The gain of every frame (`None`: they are in `each`).
    flat: Option<f64>,
    each: Vec<f64>,
}

impl Envelope {
    /// `at` for every frame.
    pub(crate) fn new(at: f64) -> Envelope {
        Envelope {
            flat: Some(at),
            each: Vec::new(),
        }
    }

    /// The gain of every frame, when it is one.
    pub(crate) fn flat(&self) -> Option<f64> {
        self.flat
    }

    /// The gain of frame `k`.
    #[cfg(test)]
    pub(crate) fn at(&self, k: usize) -> f64 {
        self.flat.unwrap_or_else(|| self.each[k])
    }

    /// As the mix takes gains.
    pub(crate) fn as_gains(&self) -> Gains<'_> {
        match self.flat {
            Some(g) => Gains::Flat(g),
            None => Gains::Each(&self.each),
        }
    }

    /// `gain`'s gains for the `n` frames from `first` at `rate`; its waiting
    /// ramp starts at `start` (a frame among them) when given.
    pub(crate) fn fill(
        &mut self,
        gain: &mut Gain,
        first: u64,
        n: usize,
        start: Option<u64>,
        rate: u32,
    ) {
        self.each.resize(n, 0.0);
        self.flat = gain.fill(first, start, rate, &mut self.each);
    }

    /// Starts a chunk of `n` frames filled in part by part: then
    /// [`Envelope::fill_part`] for each part in order, then
    /// [`Envelope::join`].
    pub(crate) fn cut(&mut self, n: usize) {
        self.each.resize(n, 0.0);
        self.flat = None;
    }

    /// `gain`'s gains for the part from frame `from` to `to` of the chunk
    /// that starts at device frame `first` (a ramp waiting for now starts
    /// on the part's first frame).
    pub(crate) fn fill_part(
        &mut self,
        gain: &mut Gain,
        first: u64,
        from: usize,
        to: usize,
        rate: u32,
    ) {
        let at = first + from as u64;
        let start = (gain.next() == Some(When::Now)).then_some(at);
        let out = &mut self.each[from..to];
        if let Some(level) = gain.fill(at, start, rate, out) {
            out.fill(level);
        }
    }

    /// The parts filled: one gain for all, when they come to one.
    pub(crate) fn join(&mut self) {
        let e = &self.each;
        self.flat = e.first().copied().filter(|&x| e.iter().all(|&y| y == x));
    }

    /// The same gains as `other`.
    pub(crate) fn assign(&mut self, other: &Envelope) -> &mut Envelope {
        self.flat = other.flat;
        if other.flat.is_none() {
            self.each.clear();
            self.each.extend_from_slice(&other.each);
        }
        self
    }

    /// Times `other`, frame by frame (over the same frames).
    pub(crate) fn times(&mut self, other: &Envelope) -> &mut Envelope {
        match (self.flat, other.flat) {
            (_, Some(1.0)) => {}
            (Some(a), Some(b)) => self.flat = Some(a * b),
            (Some(a), None) => {
                self.each.clear();
                self.each.extend(other.each.iter().map(|g| a * g));
                self.flat = None;
            }
            (None, Some(b)) => self.each.iter_mut().for_each(|x| *x *= b),
            (None, None) => {
                for (x, g) in self.each.iter_mut().zip(&other.each) {
                    *x *= g;
                }
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn each(g: &[f64]) -> Envelope {
        Envelope {
            flat: None,
            each: g.to_vec(),
        }
    }

    #[test]
    fn envelopes_multiply_flat_or_frame_by_frame() {
        let mut a = Envelope::new(0.5);
        a.times(&Envelope::new(0.5));
        assert_eq!(a.flat(), Some(0.25));
        a.times(&each(&[1.0, 0.5]));
        assert_eq!((a.flat(), &a.each[..]), (None, &[0.25, 0.125][..]));
        a.times(&Envelope::new(2.0)).times(&each(&[4.0, 2.0]));
        assert_eq!(a.each, [2.0, 0.5]);
        let mut b = Envelope::new(1.0);
        b.assign(&a).times(&Envelope::new(1.0));
        assert_eq!((b.flat(), &b.each[..]), (None, &[2.0, 0.5][..]));
        b.assign(&Envelope::new(0.0));
        assert_eq!(b.flat(), Some(0.0));
    }
}
