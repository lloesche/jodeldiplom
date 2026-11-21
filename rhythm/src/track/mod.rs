//! Trackers: from an activation [`Curve`] to beat times.
//!
//! - [`Hmm`]: a hidden Markov model over tempo and the position within the
//!   beat, decoded with Viterbi. Tempo may drift from beat to beat; jumps
//!   cost more the larger they are.
//! - [`Plp`]: the predominant local pulse: the sinusoid that best fits the
//!   curve around each moment, added up over time; its peaks are the
//!   beats.
//! - [`Peaks`]: the activation's own peaks, for activations that are
//!   already beat probabilities.

mod hmm;
mod peaks;
mod plp;

pub use hmm::Hmm;
pub use peaks::Peaks;
pub use plp::Plp;

use crate::activation::Curve;

/// Something that finds beats in an activation.
pub trait Tracker: Send + Sync {
    /// A short name for reports, such as `hmm`.
    fn name(&self) -> &'static str;
    /// The beat times in seconds, ascending.
    fn track(&self, curve: &Curve) -> Vec<f64>;
}

/// The frame of the largest value in `values[lo..hi]`, refined between
/// frames by fitting a parabola through it and its neighbours.
pub(crate) fn peak_in(values: &[f32], lo: usize, hi: usize) -> Option<f64> {
    let hi = hi.min(values.len());
    if lo >= hi {
        return None;
    }
    let best = (lo..hi).max_by(|&a, &b| values[a].total_cmp(&values[b]))?;
    Some(refine(values, best))
}

/// `frame` moved to the vertex of the parabola through it and its
/// neighbours (at most half a frame).
pub(crate) fn refine(values: &[f32], frame: usize) -> f64 {
    if frame == 0 || frame + 1 >= values.len() {
        return frame as f64;
    }
    let (a, b, c) = (
        f64::from(values[frame - 1]),
        f64::from(values[frame]),
        f64::from(values[frame + 1]),
    );
    let denominator = a - 2.0 * b + c;
    if denominator >= 0.0 {
        return frame as f64;
    }
    frame as f64 + (0.5 * (a - c) / denominator).clamp(-0.5, 0.5)
}
