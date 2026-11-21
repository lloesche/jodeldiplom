//! A beat wherever the activation peaks above one half.
//!
//! For activations that already are beat probabilities (the network's),
//! this is all the tracking they need: a frame is a beat when it is the
//! largest within ±70 ms and above 0.5 (and not the first or last
//! frame); peaks a frame apart are one beat
//! at their mean, and each is refined between frames. It follows the
//! music exactly, tempo changes and missing beats included, with no idea
//! of a steady pulse; the other trackers supply that.

use super::{refine, Tracker};
use crate::activation::Curve;

/// The peak picker's settings.
#[derive(Clone, Copy, Debug)]
pub struct Peaks {
    /// The smallest activation that is a beat.
    pub threshold: f32,
    /// A peak is the largest within this many seconds either side.
    pub window: f64,
}

impl Default for Peaks {
    fn default() -> Peaks {
        Peaks {
            threshold: 0.5,
            window: 0.07,
        }
    }
}

impl Tracker for Peaks {
    fn name(&self) -> &'static str {
        "peaks"
    }

    fn track(&self, curve: &Curve) -> Vec<f64> {
        let v = &curve.values;
        let reach = (self.window * curve.fps).round().max(1.0) as usize;
        // A peak needs a frame on both sides: at the ends the curve may
        // still be rising.
        let mut frames: Vec<usize> = (1..v.len().saturating_sub(1))
            .filter(|&i| {
                let lo = i.saturating_sub(reach);
                let hi = (i + reach + 1).min(v.len());
                v[i] > self.threshold && v[lo..hi].iter().all(|&x| x <= v[i])
            })
            .collect();
        // Neighbouring frames of one flat top: keep the first, at their mean.
        let mut beats = Vec::with_capacity(frames.len());
        frames.dedup_by(|next, kept| *next <= *kept + 1);
        for f in frames {
            let at = if v.get(f + 1) == Some(&v[f]) {
                f as f64 + 0.5
            } else {
                refine(v, f)
            };
            beats.push(at / curve.fps);
        }
        beats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaks_above_one_half() {
        let fps = 50.0;
        let mut values = vec![0.01f32; 500];
        for b in [50, 75, 100, 125] {
            values[b - 1] = 0.4;
            values[b] = 0.9;
            values[b + 1] = 0.4;
        }
        values[300] = 0.45; // too weak
        values[400] = 0.8; // a flat top of two frames
        values[401] = 0.8;
        let beats = Peaks::default().track(&Curve { values, fps });
        let want = [1.0, 1.5, 2.0, 2.5, 8.01];
        assert_eq!(beats.len(), want.len(), "{beats:?}");
        for (b, w) in beats.iter().zip(want) {
            assert!((b - w).abs() < 1e-9, "{beats:?}");
        }
    }
}
