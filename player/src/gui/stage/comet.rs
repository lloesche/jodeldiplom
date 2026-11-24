//! The singer's arrow's comet tail: a glow tapering back along the path
//! the arrow took (its places as the lane carries them away), so it curves
//! as the voice went, with stardust sprinkled along it (sparks thrown off
//! the arrow slowly, [`DUST`] a second, carried away with the lane).

use std::collections::VecDeque;

use super::{disc, rgba, Instance, Lane, Rgb, WHITE};

/// How long a tail the comet has: the arrow's places this far back.
const LONG: f64 = 0.4;
/// Stardust off the arrow a second.
pub const DUST: f32 = 90.0;

/// The arrow's places lately: when (the song's time on the lane) and how
/// far down.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Comet {
    path: VecDeque<(f64, f32)>,
}

impl Comet {
    /// The arrow at `at` (when, how far down; none: not shown, the tail
    /// gone).
    pub fn step(&mut self, at: Option<(f64, f32)>) {
        let Some((now, y)) = at else {
            self.path.clear();
            return;
        };
        // A jump back (a seek): a tail afresh.
        if self.path.back().is_some_and(|&(t, _)| t > now) {
            self.path.clear();
        }
        self.path.push_back((now, y));
        while self.path.front().is_some_and(|&(t, _)| now - t > LONG) {
            self.path.pop_front();
        }
    }

    /// The tail into `out` on `lane`: in `color`, for pills `p` high, as
    /// much as the arrow shows (`shown`). Thickest at the arrow, thinning
    /// and fading back along its path.
    pub fn draw(&self, out: &mut Vec<Instance>, lane: &Lane, (color, p): (Rgb, f32), shown: f32) {
        let Some(&(now, _)) = self.path.back() else {
            return;
        };
        for &(t, y) in &self.path {
            let left = 1.0 - ((now - t) / LONG) as f32;
            if left <= 0.0 {
                continue;
            }
            let r = p * 0.1 * left.powf(0.7);
            let core = mix(WHITE, color, 1.0 - left);
            out.push(disc(
                (lane.x(t), y),
                r,
                rgba(core, 0.55 * shown * left.powf(1.4)),
                rgba(color, 0.22 * shown * left),
                p * 0.25 * left,
            ));
        }
    }
}

fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tail keeps the arrow's last places, no older than its length,
    /// and starts afresh after a jump back or when the arrow goes.
    #[test]
    fn the_tail_follows_the_arrow_back_its_length() {
        let mut c = Comet::default();
        for k in 0..60 {
            c.step(Some((f64::from(k) / 60.0, k as f32)));
        }
        let first = c.path.front().unwrap().0;
        assert!((59.0 / 60.0 - first) <= LONG + 1e-9, "{first}");
        assert_eq!(c.path.back(), Some(&(59.0 / 60.0, 59.0)));
        c.step(Some((0.2, 1.0)));
        assert_eq!(c.path.len(), 1);
        c.step(None);
        assert!(c.path.is_empty());
    }
}
