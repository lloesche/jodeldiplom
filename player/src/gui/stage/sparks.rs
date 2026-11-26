//! The sparks off the singer's arrow ([`Sparks`]): stardust along its
//! comet's tail all the while it shows, and a shower of sparks while it is
//! on the note, each owed by the song's time gone by ([`Sparks::fly`]);
//! each spark flies its own way and by with the lane, slowing, turning and
//! twinkling, and is gone at the end of its life ([`Sparks::step`],
//! [`Sparks::draw`]). At most [`MOST`] at once: the oldest go first.

use std::collections::VecDeque;

use super::super::sparks::noise;
use super::super::stage_gpu::{Instance, Shape};
use super::comet::DUST;
use super::palette::{rgba, Rgb};

/// The most sparks at once.
const MOST: usize = 700;
/// Sparks a second off the arrow on the note.
const ON_NOTE: f32 = 40.0;

/// A spark flying off the arrow.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Spark {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    age: f32,
    life: f32,
    size: f32,
    turn: f32,
    spin: f32,
    color: Rgb,
    phase: f32,
}

/// The sparks flying, and what is owed of them.
#[derive(Default)]
pub(super) struct Sparks {
    list: VecDeque<Spark>,
    /// The last number drawn for them ([`noise`]'s).
    seed: u32,
    /// Sparks owed to the next frame (a share of one).
    owed: f32,
    /// Stardust owed to the next frame.
    dust: f32,
}

impl Sparks {
    fn rand(&mut self) -> f32 {
        self.seed = self.seed.wrapping_add(1);
        noise(self.seed)
    }

    /// `n` sparks off (`x`, `y`): `color`, `size` across, `spread` how
    /// far they fly from there, `reach` pixels a second at 1.
    fn emit(
        &mut self,
        n: usize,
        (x, y): (f32, f32),
        color: Rgb,
        size: f32,
        spread: f32,
        reach: f32,
    ) {
        for _ in 0..n {
            if self.list.len() >= MOST {
                self.list.pop_front();
            }
            let (a, b, c, d, e) = (
                self.rand(),
                self.rand(),
                self.rand(),
                self.rand(),
                self.rand(),
            );
            let white = self.rand() < 0.3;
            let (f, g, i, j) = (self.rand(), self.rand(), self.rand(), self.rand());
            self.list.push_back(Spark {
                x: x + (a - 0.5) * size * 0.6,
                y: y + (b - 0.5) * size * 0.6,
                vx: (c - 0.65) * reach * 4.0 * spread,
                vy: (d - 0.5) * reach * 4.0 * spread,
                age: 0.0,
                life: 0.4 + 0.7 * e,
                size: size * (0.35 + 0.65 * f),
                turn: g * std::f32::consts::TAU,
                spin: (i - 0.5) * 6.0,
                color: if white { [1.0, 0.97, 0.88] } else { color },
                phase: j * std::f32::consts::TAU,
            });
        }
    }

    /// What `dt` seconds of the song owe off the arrow `at` (none where it
    /// hardly shows: nothing is owed then) in `color`, the pills `p` high:
    /// its stardust, and its sparks where it is `hitting` the note.
    pub(super) fn fly(
        &mut self,
        at: Option<(f32, f32)>,
        color: Rgb,
        p: f32,
        hitting: bool,
        dt: f32,
    ) {
        let Some(at) = at else {
            (self.owed, self.dust) = (0.0, 0.0);
            return;
        };
        self.dust += DUST * dt;
        let n = self.dust.floor() as usize;
        self.dust -= n as f32;
        self.emit(n, at, color, p * 0.26, 0.12, p);
        self.owed += if hitting { ON_NOTE } else { 0.0 } * dt;
        let n = self.owed.floor() as usize;
        self.owed -= n as f32;
        self.emit(n, at, color, p * 0.5, 1.0, p);
    }

    /// Each spark `dt` seconds of the song on: its own way, and by with
    /// the lane (`pps` pixels a second); those at the end of their life
    /// gone.
    pub(super) fn step(&mut self, dt: f32, pps: f32) {
        for s in &mut self.list {
            s.age += dt;
            s.x += (s.vx - pps) * dt;
            s.y += s.vy * dt;
            let drag = (-2.4 * dt).exp();
            s.vx *= drag;
            s.vy *= drag;
            s.turn += s.spin * dt;
        }
        self.list.retain(|s| s.age < s.life);
    }

    /// The sparks into `out`, twinkling, shrinking and fading as they age.
    pub(super) fn draw(&self, out: &mut Vec<Instance>) {
        for s in &self.list {
            let left = 1.0 - s.age / s.life;
            let twinkle = 0.6 + 0.4 * (s.age * 26.0 + s.phase).sin();
            let size = s.size * (0.45 + 0.55 * left);
            let m = size + 1.0;
            let quad = [s.x - m, s.y - m, s.x + m, s.y + m];
            let spark = Instance::of(Shape::Spark, quad, [s.x, s.y, size, s.turn]);
            out.push(spark.colors(rgba(s.color, left.powf(1.2) * twinkle), [0.0; 4]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::stage::palette::GOLD;

    #[test]
    fn sparks_fly_and_go() {
        let mut sparks = Sparks::default();
        sparks.emit(10, (100.0, 100.0), GOLD, 20.0, 1.0, 40.0);
        assert_eq!(sparks.list.len(), 10);
        // Most fly back, and all go by with the lane.
        assert!(sparks.list.iter().filter(|s| s.vx < 0.0).count() >= 5);
        let mut drawn = Vec::new();
        sparks.draw(&mut drawn);
        assert_eq!(drawn.len(), 10);
        sparks.step(1.2, 100.0);
        assert!(sparks.list.is_empty());
    }
}
