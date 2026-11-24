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

    /// Room for the gains of `n` frames, so filling them never allocates.
    pub(crate) fn reserve(&mut self, n: usize) {
        self.each.reserve(n.saturating_sub(self.each.len()));
    }

    /// The gain of every frame, when it is one.
    pub(crate) fn flat(&self) -> Option<f64> {
        self.flat
    }

    /// The gain of frame `k`.
    pub(crate) fn at(&self, k: usize) -> f64 {
        self.flat.unwrap_or_else(|| self.each[k])
    }

    /// `at` for every frame, its buffer kept.
    pub(crate) fn set_flat(&mut self, at: f64) {
        self.flat = Some(at);
    }

    /// The gains of `n` frames, frame `k`'s `gain(k)` (in order): one for
    /// all, when they come to one.
    pub(crate) fn fill_with(&mut self, n: usize, mut gain: impl FnMut(usize) -> f64) {
        self.each.resize(n, 0.0);
        for (k, g) in self.each.iter_mut().enumerate() {
            *g = gain(k);
        }
        self.join();
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

    /// One less `other`'s gains (`1 − g`, frame by frame; exactly 1 where
    /// they are 0).
    pub(crate) fn assign_complement(&mut self, other: &Envelope) -> &mut Envelope {
        self.flat = other.flat.map(|g| 1.0 - g);
        if other.flat.is_none() {
            self.each.clear();
            self.each.extend(other.each.iter().map(|g| 1.0 - g));
        }
        self
    }

    /// The gain of the last frame (`n` frames in the chunk).
    pub(crate) fn last(&self, n: usize) -> f64 {
        self.flat
            .unwrap_or_else(|| self.each.get(n.wrapping_sub(1)).copied().unwrap_or(0.0))
    }

    /// Times `other`, frame by frame (over the same frames). Silence times
    /// anything stays silence, for all of the chunk: a lane at 0 is not
    /// heard, whatever ramp runs on it.
    pub(crate) fn times(&mut self, other: &Envelope) -> &mut Envelope {
        match (self.flat, other.flat) {
            (_, Some(1.0)) | (Some(0.0), _) => {}
            (_, Some(0.0)) => self.flat = Some(0.0),
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
        // Silence stays flat, whatever it is multiplied by.
        b.times(&each(&[0.5, 0.25]));
        assert_eq!(b.flat(), Some(0.0));
        let mut c = each(&[0.5, 0.25]);
        c.times(&Envelope::new(0.0));
        assert_eq!(c.flat(), Some(0.0));
        c.assign_complement(&each(&[0.25, 1.0]));
        assert_eq!((c.flat(), c.last(2)), (None, 0.0));
    }
}
