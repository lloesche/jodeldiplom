//! A filter of the canceller: a partition per [`BLOCK`] of its window,
//! newest first, each a set of bins per reference channel; how it predicts
//! the echo, and how it learns.

use super::fft::{Bins, Fft, BINS, C32};
use super::reference::{Frame, History, CHANNELS, SILENT};
use super::BLOCK;

/// A filter's partitions.
pub(super) type Filter = Vec<Frame>;

/// The filter's step size while it learns.
const STEP: f32 = 0.6;
/// How many times larger than the whitened reference's the reference's
/// own amplitude may make a step (a step is STEP times this at most, under
/// the 2 beyond which it grows without end). On five real recordings from a
/// laptop the filter that learns had grown beyond numbers within a song
/// without it; at 2 the canceller learnt too slowly (9 of its tests failed),
/// at 4 and 6 it grew again.
const WHITE_GAIN_MOST: f32 = 3.0;
/// Partitions whose impulse response is constrained per block.
const CONSTRAINED: usize = 8;

/// A filter of `partitions` that predicts nothing.
pub(super) fn silent(partitions: usize) -> Filter {
    vec![SILENT; partitions]
}

/// Forgets all `filter` learnt.
pub(super) fn clear(filter: &mut Filter) {
    filter.fill(SILENT);
}

/// `filter` scaled by `gain`.
pub(super) fn scale(filter: &mut Filter, gain: f32) {
    for c in filter.iter_mut().flatten().flatten() {
        *c = c.scale(gain);
    }
}

/// The echo `filter` predicts from the reference's history, the window
/// starting `offset` blocks back: the time samples of the transform, the
/// newer half the block's.
pub(super) fn predict(
    fft: &Fft,
    filter: &[Frame],
    history: &History,
    offset: usize,
) -> [f32; 2 * BLOCK] {
    let mut y = [C32::default(); BINS];
    for (p, w) in filter.iter().enumerate() {
        let x = history.at(offset + p);
        for c in 0..CHANNELS {
            for k in 0..BINS {
                y[k] = y[k].add(w[c][k].mul(x[c][k]));
            }
        }
    }
    fft.inverse(&y)
}

/// The partition where `filter` has the most (the echo's loudest part).
pub(super) fn peak(filter: &[Frame]) -> usize {
    let energy = |w: &Frame| w.iter().flatten().map(|c| c.norm()).sum::<f32>();
    filter
        .iter()
        .map(energy)
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(p, _)| p)
}

/// `bins`' impulse response kept to its first half: a linear, not a
/// circular, convolution.
fn constrain_one(fft: &Fft, bins: &mut Bins) {
    let mut t = fft.inverse(bins);
    for v in &mut t[BLOCK..] {
        *v = 0.0;
    }
    *bins = fft.real(&t[..BLOCK], &t[BLOCK..]);
}

/// Every partition of `filter` constrained ([`constrain_one`]).
pub(super) fn constrain(fft: &Fft, filter: &mut [Frame]) {
    for frame in filter.iter_mut() {
        for bins in frame.iter_mut() {
            constrain_one(fft, bins);
        }
    }
}

/// The reference's histories a filter learns from: as played (what it
/// acts on) and whitened ([`super::reference::Whitener`]: what it steps
/// along), the window starting `offset` blocks back.
pub(super) struct Along<'a> {
    pub(super) plain: &'a History,
    pub(super) white: &'a History,
    pub(super) offset: usize,
}

/// One step of `filter` towards the error `error` it left, along the
/// whitened reference; `turn` the partition constrained next.
///
/// Each bin's step is normalised by the whitened reference's power over
/// the window, across its channels (the inverse of their 2 × 2 power
/// matrix: what the two play alike and apart each learn at full pace,
/// whichever is the louder, and a song panned hard, where the two are one,
/// learns as a single channel would). The diagonal is never less than the
/// reference's own power over [`WHITE_GAIN_MOST`] squared: the filter acts
/// on the reference as it is and learns along it whitened; where whitening
/// takes most of a bin away (a held bass note, most of a song's power,
/// which a laptop does not even play), a step by the whitened power alone
/// is that many times too large, and grew the filter without end.
pub(super) fn adapt(
    fft: &Fft,
    filter: &mut [Frame],
    error: &[f32; BLOCK],
    along: &Along,
    turn: &mut usize,
) {
    let e = fft.real(&[0.0; BLOCK], error);
    let mut q = [[0.0f32; BINS]; CHANNELS];
    let mut cross = [C32::default(); BINS];
    let mut plain = [[0.0f32; BINS]; CHANNELS];
    for p in 0..filter.len() {
        let x = along.white.at(along.offset + p);
        let xp = along.plain.at(along.offset + p);
        for k in 0..BINS {
            for c in 0..CHANNELS {
                q[c][k] += x[c][k].norm();
                plain[c][k] += xp[c][k].norm();
            }
            cross[k] = cross[k].add(x[0][k].conj_mul(x[1][k]));
        }
    }
    let mut total = 0.0f32;
    for c in 0..CHANNELS {
        for k in 0..BINS {
            q[c][k] = q[c][k].max(plain[c][k] / (WHITE_GAIN_MOST * WHITE_GAIN_MOST));
            total += q[c][k];
        }
    }
    let floor = total / BINS as f32 * 1e-5 + 1e-10;
    // Each bin's inverse: [[q11, q12], [conj q12, q22]]⁻¹, scaled by STEP,
    // a little added to the diagonal so it always is one (a channel
    // silent, or the two one: a song panned hard).
    let mut inverse = [(0.0f32, 0.0f32, C32::default()); BINS];
    for k in 0..BINS {
        let more = floor + 1e-3 * (q[0][k] + q[1][k]);
        let (a, b) = (q[0][k] + more, q[1][k] + more);
        let det = a * b - cross[k].norm();
        let s = STEP / det;
        inverse[k] = (b * s, a * s, cross[k].scale(-s));
    }
    for (p, w) in filter.iter_mut().enumerate() {
        let x = along.white.at(along.offset + p);
        for k in 0..BINS {
            let (i11, i22, i12) = inverse[k];
            let g0 = x[0][k].conj_mul(e[k]);
            let g1 = x[1][k].conj_mul(e[k]);
            // i21 is conj(i12).
            let i21 = C32 {
                re: i12.re,
                im: -i12.im,
            };
            w[0][k] = w[0][k].add(g0.scale(i11).add(i12.mul(g1)));
            w[1][k] = w[1][k].add(g1.scale(i22).add(i21.mul(g0)));
        }
    }
    // Some partitions constrained per block, in turn, so wrapped errors
    // never build up; what has not been yet fits the music it learnt from,
    // not a voice that starts and stops.
    for _ in 0..CONSTRAINED.min(filter.len()) {
        let p = *turn % filter.len();
        *turn = turn.wrapping_add(1);
        for bins in filter[p].iter_mut() {
            constrain_one(fft, bins);
        }
    }
}
