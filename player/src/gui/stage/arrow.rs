//! The singer's arrow on the stage: gliding to the pitch heard, on a
//! critically damped spring towards the middle of the pitches heard
//! lately, fading where there is none.

/// How fast the arrow follows the pitch: a weight on a rubber band, its
/// natural frequency (radians a second), damped just enough never to swing
/// past (critically). Most of the way to a new pitch in about a tenth of a
/// second.
const SPRING: f32 = 45.0;
/// What the arrow goes for: the middle (median) of the pitches heard this
/// long back, so a stray one never pulls it.
const STEADY: f32 = 0.035;

/// The singer's arrow: where it is (pixels down) and how fast it moves,
/// and how much it shows; the pitches it was given lately (how long ago,
/// where), to go for their middle.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Arrow {
    pub(super) y: f32,
    speed: f32,
    pub(super) shown: f32,
    placed: bool,
    lately: std::collections::VecDeque<(f32, f32)>,
}

impl Arrow {
    /// `dt` seconds on, gliding to `to` (none: fading).
    pub(super) fn step(&mut self, to: Option<f32>, dt: f32) {
        for heard in &mut self.lately {
            heard.0 += dt;
        }
        self.lately.retain(|&(ago, _)| ago <= STEADY);
        match to {
            Some(to) => {
                self.lately.push_back((0.0, to));
                let to = self.steady();
                if !self.placed {
                    self.y = to;
                    self.speed = 0.0;
                    self.placed = true;
                }
                // A critically damped spring, in steps short enough to move
                // alike at any frame rate.
                let steps = (dt / 0.004).ceil().max(1.0) as usize;
                let h = dt / steps as f32;
                for _ in 0..steps {
                    let pull = -2.0 * SPRING * self.speed - SPRING * SPRING * (self.y - to);
                    self.speed += pull * h;
                    self.y += self.speed * h;
                }
                self.shown = (self.shown + dt / 0.08).min(1.0);
            }
            None => {
                self.shown = (self.shown - dt / 0.15).max(0.0);
                if self.shown == 0.0 {
                    self.placed = false;
                    self.lately.clear();
                }
            }
        }
    }

    /// The middle of the pitches given lately; where they are even, of the
    /// two middle ones the nearer to where the arrow is (a new pitch moves
    /// it only once it is most of those heard).
    fn steady(&self) -> f32 {
        let mut ys: Vec<f32> = self.lately.iter().map(|&(_, y)| y).collect();
        ys.sort_by(f32::total_cmp);
        let upper = ys[ys.len() / 2];
        if ys.len() % 2 == 1 || !self.placed {
            return upper;
        }
        let lower = ys[ys.len() / 2 - 1];
        if (lower - self.y).abs() < (upper - self.y).abs() {
            lower
        } else {
            upper
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_arrow_glides_to_the_pitch_alike_at_any_frame_rate() {
        let mut ends = Vec::new();
        for fps in [60.0f32, 144.0] {
            let mut arrow = Arrow::default();
            arrow.step(Some(100.0), 1.0 / fps);
            let mut ys = Vec::new();
            for _ in 0..(0.5 * fps).round() as usize {
                arrow.step(Some(200.0), 1.0 / fps);
                ys.push(arrow.y);
            }
            let at = |s: f32| ys[(s * fps).round() as usize - 1];
            // Put for its first moments (the new pitch not yet the middle
            // of those heard), most of the way within 0.3 s, never past it,
            // and still by half a second.
            assert!(at(0.01) < 110.0, "{fps}: {}", at(0.01));
            assert!(at(0.3) > 190.0, "{fps}: {}", at(0.3));
            let most = ys.iter().copied().fold(0.0, f32::max);
            assert!(most <= 200.01, "{fps}: {most}");
            assert!((at(0.5) - 200.0).abs() < 0.5, "{fps}: {}", at(0.5));
            ends.push(at(0.3));
        }
        assert!((ends[0] - ends[1]).abs() < 3.0, "{ends:?}");
        // Stray pitches, a few in a row, do not move it at all.
        let mut arrow = Arrow::default();
        for _ in 0..20 {
            arrow.step(Some(100.0), 0.004);
        }
        for _ in 0..3 {
            arrow.step(Some(300.0), 0.004);
        }
        let mut most: f32 = 0.0;
        for _ in 0..50 {
            arrow.step(Some(100.0), 0.004);
            most = most.max((arrow.y - 100.0).abs());
        }
        assert!(most < 0.01, "{most}");
        // Without a pitch it fades, and starts afresh where the next is.
        let mut arrow = Arrow::default();
        arrow.step(Some(100.0), 0.016);
        for _ in 0..20 {
            arrow.step(None, 0.016);
        }
        assert_eq!(arrow.shown, 0.0);
        arrow.step(Some(300.0), 0.016);
        assert_eq!(arrow.y, 300.0);
    }
}
