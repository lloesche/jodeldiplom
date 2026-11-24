//! Finding how late the echo comes, before the filter has learnt it.

use super::fft::{Bins, C32};

/// The bins the finder listens to (about 375 Hz to 9 kHz at 48 kHz):
/// where every speaker plays and a room is still quiet.
const LOW: usize = 2;
const HIGH: usize = 48;
const WIDTH: usize = HIGH - LOW;
/// How fast the products follow, per block (about a second).
const FOLLOW: f32 = 1.0 / 250.0;
/// Blocks before the delay that stands out that its own frames reach (a
/// frame is two blocks), and after it that the room's tail may (about
/// 100 ms).
const NEAR: usize = 2;
const TAIL_AFTER: usize = 40;
/// How far over every delay but those one must stand out.
const OVER_OTHERS: f32 = 3.0;
/// How far over what chance leaves (the products of sounds with nothing in
/// common; the most of 400 delays of that is about 1.5 times it). In a
/// living room whose tail is louder than the speaker's own sound the delay
/// stands about 3 times over it.
const OVER_CHANCE: f32 = 2.5;
/// Blocks seen with the reference playing before the finder says anything
/// (about 2 s of music).
pub(super) const LEARN_FIRST: u32 = 750;
/// Looks running the same delay must stand out in.
const LOOKS: u32 = 4;

/// Finds how late the echo comes: the microphone's bins against the
/// reference's of each block before, up to [`super::MAX_DELAY`], their
/// phases only (each bin of either made unit size). At the echo's delay the
/// two turn alike bin by bin, block after block; at any other they do not,
/// however alike the music's loudness moves there (a song's beat repeats
/// its loudness every bar, and matched by loudness the delay was a beat
/// off as often as not). A speaker's distortion and what the microphone
/// hears besides only weaken it.
pub(super) struct DelayFinder {
    /// The reference's bins (phase only), a ring.
    reference: Vec<[C32; WIDTH]>,
    head: usize,
    /// Per delay in blocks: per bin, the smoothed product of the
    /// microphone's with the reference's that many blocks before.
    products: Vec<[C32; WIDTH]>,
    /// Blocks seen with the reference playing.
    seen: u32,
    /// The delay that stood out at the last looks, and in how many
    /// running.
    candidate: Option<(usize, u32)>,
}

/// `x`'s bins the finder hears, each of unit size (none: silent).
fn phases(x: &Bins) -> [C32; WIDTH] {
    let mut out = [C32::default(); WIDTH];
    for (o, c) in out.iter_mut().zip(&x[LOW..HIGH]) {
        let n = c.norm();
        if n > 1e-18 {
            *o = c.scale(1.0 / n.sqrt());
        }
    }
    out
}

impl DelayFinder {
    pub(super) fn new(lags: usize) -> DelayFinder {
        DelayFinder {
            reference: vec![[C32::default(); WIDTH]; lags],
            head: 0,
            products: vec![[C32::default(); WIDTH]; lags],
            seen: 0,
            candidate: None,
        }
    }

    pub(super) fn clear(&mut self) {
        self.reference.fill([C32::default(); WIDTH]);
        self.products.fill([C32::default(); WIDTH]);
        self.seen = 0;
        self.candidate = None;
    }

    /// One block of the reference and the microphone (as bins).
    pub(super) fn push(&mut self, reference: &Bins, mic: &Bins) {
        let lags = self.reference.len();
        self.head = (self.head + 1) % lags;
        self.reference[self.head] = phases(reference);
        let m = phases(mic);
        for (lag, p) in self.products.iter_mut().enumerate() {
            let old = &self.reference[(self.head + lags - lag) % lags];
            for k in 0..WIDTH {
                let g = old[k].conj_mul(m[k]);
                p[k] = p[k].add(g.sub(p[k]).scale(FOLLOW));
            }
        }
        self.seen = self.seen.saturating_add(1);
    }

    /// Looks at the delays: one that stands out (far over any other, the
    /// room's tail after it aside, and over chance) the same in a few looks
    /// running is the echo's.
    pub(super) fn look(&mut self) {
        let score = |p: &[C32; WIDTH]| p.iter().map(|c| c.norm()).sum::<f32>();
        let scores = self.products.iter().map(score);
        let best = scores
            .clone()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1));
        // What chance leaves: the products of unit phasors with nothing in
        // common, smoothed, are about FOLLOW / 2 a bin.
        let chance = WIDTH as f32 * FOLLOW / 2.0;
        let candidate = best.filter(|&(best, top)| {
            let other = scores
                .clone()
                .enumerate()
                .filter(|(l, _)| *l + NEAR < best || *l > best + TAIL_AFTER)
                .map(|(_, s)| s)
                .fold(0.0f32, f32::max);
            top > OVER_CHANCE * chance && top > OVER_OTHERS * other
        });
        match (candidate, self.candidate) {
            (Some((l, _)), Some((was, n))) if l.abs_diff(was) <= 1 => {
                self.candidate = Some((l, n + 1));
            }
            (Some((l, _)), _) => self.candidate = Some((l, 1)),
            (None, _) => self.candidate = None,
        }
    }

    /// The delay in blocks, once it has stood out a few looks running.
    pub(super) fn lag(&self) -> Option<usize> {
        if self.seen < LEARN_FIRST {
            return None;
        }
        self.candidate.filter(|c| c.1 >= LOOKS).map(|c| c.0)
    }
}
