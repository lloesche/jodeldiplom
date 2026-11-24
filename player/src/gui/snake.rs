//! A word's bubble on the stage as a tube ([`Snake`]): one smooth tube of
//! constant radius with round ends, its middle line running through its
//! notes, each note's stretch at its own pitch and a smooth S from one
//! pitch to the next; a word on one pitch a plain capsule. Its middle line
//! is a polyline, sorted across, handed to the GPU in the frame's path
//! buffer ([`super::stage_gpu::StageFrame::paths`]); the shader
//! (`stage.wgsl`) draws the tube as a signed distance from it, and the
//! liquid along it, by where
//! along the tube a pixel is (its nearest point's place across, so the
//! liquid's front stands square to the tube wherever it bends).
//!
//! [`nearest`] is the shader's `tube_near` on the processor, its reference
//! (held to it on the GPU, `super::shader_check`).

use super::liquid::{amplitude, GROW};
use super::stage_gpu::{kind, Instance, Pour, Skin, Strain};

/// How much of a bubble's bulge its middle line takes (it grows along and
/// up and down by this share of it; its thickness by all of it).
pub const PATH_GROW: f32 = 0.35;
/// The points of an S from one pitch to the next, between its ends.
const S_POINTS: usize = 8;
/// The most of a note an S to or from it takes, each way: under a half,
/// so that the S's either side of a note never meet (a word's notes lie
/// end to end, [`crate::core::singing::notes_of`]) and each note keeps a
/// stretch at its own pitch.
const S_SHARE: f32 = 0.4;
const _: () = assert!(S_SHARE < 0.5);

/// A tube: its middle line (points sorted across), and its radius.
#[derive(Clone, Debug, PartialEq)]
pub struct Snake {
    /// Its middle line.
    pub points: Vec<(f32, f32)>,
    /// Its radius.
    pub radius: f32,
}

/// `smoothstep` from 0 to 1.
fn ease(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Snake {
    /// The tube from `left` to `right` across (its round ends included),
    /// `radius` thick each way, through `notes` (each from, to across and
    /// how far down its middle is), in order.
    pub fn through(notes: &[(f32, f32, f32)], (left, right): (f32, f32), radius: f32) -> Snake {
        let (from, to) = (left + radius, right - radius);
        let mid = (left + right) / 2.0;
        let (from, to) = if from < to { (from, to) } else { (mid, mid) };
        // Points along each S between two notes on different pitches (a
        // straight run needs none between its ends).
        let mut xs = vec![from, to];
        for pair in notes.windows(2) {
            let (at, wide) = ess(pair[0], pair[1], radius);
            if (pair[1].2 - pair[0].2).abs() < 0.25 {
                continue;
            }
            let (a, b) = (at - wide, at + wide);
            for k in 0..=S_POINTS + 1 {
                let x = a + (b - a) * k as f32 / (S_POINTS + 1) as f32;
                if x > from && x < to {
                    xs.push(x);
                }
            }
        }
        xs.sort_by(f32::total_cmp);
        xs.dedup_by(|a, b| (*a - *b).abs() < 0.25);
        let mut snake = Snake {
            points: Vec::with_capacity(xs.len().max(2)),
            radius,
        };
        let height = |x: f32| y_at(notes, x, radius);
        snake.points = xs.iter().map(|&x| (x, height(x))).collect();
        if snake.points.len() == 1 {
            snake.points.push(snake.points[0]);
        }
        snake
    }

    /// How far down its middle line is at `x` (its ends' height past them).
    pub fn y_at(&self, x: f32) -> f32 {
        let p = &self.points;
        let i = p.partition_point(|q| q.0 < x);
        if i == 0 {
            return p[0].1;
        }
        let Some(&b) = p.get(i) else {
            return p[p.len() - 1].1;
        };
        let a = p[i - 1];
        let t = (x - a.0) / (b.0 - a.0).max(1e-6);
        a.1 + (b.1 - a.1) * t
    }

    /// Its middle: half way across, half way between its highest and its
    /// lowest; what it bulges round.
    pub fn centre(&self) -> (f32, f32) {
        let (lo, hi) = self.span_y();
        let (a, b) = (self.points[0].0, self.points[self.points.len() - 1].0);
        ((a + b) / 2.0, (lo + hi) / 2.0)
    }

    fn span_y(&self) -> (f32, f32) {
        (self.points.iter()).fold((f32::MAX, f32::MIN), |m, p| (m.0.min(p.1), m.1.max(p.1)))
    }

    /// Where text `wide` across reads best along it between `from` and
    /// `to` (its left): the flattest stretch that wide (a note's own pitch
    /// where one is long enough), of those as flat the nearest the middle.
    pub fn flattest(&self, wide: f32, (from, to): (f32, f32)) -> f32 {
        let middle = (from + to - wide) / 2.0;
        if to - from <= wide {
            return middle;
        }
        const STEP: f32 = 3.0;
        let bend = |x: f32| {
            let n = (wide / STEP).ceil() as usize;
            let ys = (0..=n).map(|k| self.y_at(x + wide * k as f32 / n.max(1) as f32));
            let (lo, hi) = ys.fold((f32::MAX, f32::MIN), |m, y| (m.0.min(y), m.1.max(y)));
            hi - lo
        };
        let starts = ((to - from - wide) / STEP).ceil() as usize;
        (0..=starts)
            .map(|k| (from + k as f32 * STEP).min(to - wide))
            .map(|x| (bend(x) + 0.02 * (x - middle).abs(), x))
            .fold(
                (f32::MAX, middle),
                |best, c| if c.0 < best.0 { c } else { best },
            )
            .1
    }

    /// The quad a shape of it covers between `x0` and `x1` across (all of
    /// it: unbounded), reaching `more` past its bulged skin.
    fn quad(&self, (x0, x1): (f32, f32), more: f32) -> [f32; 4] {
        let c = self.centre();
        let grow = 1.0 + GROW * PATH_GROW;
        let (lo, hi) = self.span_y();
        let (a, b) = (self.points[0].0, self.points[self.points.len() - 1].0);
        let reach = self.radius * (1.0 + GROW) + more + 1.0;
        let (a, b) = (a.max(x0 - reach), b.min(x1 + reach));
        let out = |v: f32, mid: f32| mid + (v - mid) * grow;
        [
            out(a, c.0) - reach,
            out(lo, c.1) - reach,
            out(b, c.0) + reach,
            out(hi, c.1) + reach,
        ]
    }

    /// Its points into `paths` (the frame's path buffer): where they start
    /// there.
    pub fn put(&self, paths: &mut Vec<[f32; 4]>) -> u32 {
        let first = paths.len() as u32;
        paths.extend(self.points.iter().map(|&(x, y)| [x, y, 0.0, 0.0]));
        first
    }

    /// Its skin, its points at `first` in the path buffer, straining as
    /// `strain` says.
    pub fn skin(&self, first: u32, skin: &Skin, strain: &Strain) -> Instance {
        let c = self.centre();
        Instance {
            quad: self.quad((f32::MIN, f32::MAX), skin.glow),
            shape: [c.0, c.1, self.radius, first as f32],
            color: skin.body,
            color2: skin.edge,
            params: [
                kind::TUBE,
                skin.glow,
                skin.outline,
                self.points.len() as f32,
            ],
            uv: [0.0; 4],
            strain: Instance::strain_of(strain, false),
        }
    }

    /// Its liquid poured as `pour` says (its span along it, across),
    /// straining with it; none where too little shows.
    pub fn liquid(&self, first: u32, pour: &Pour, strain: &Strain) -> Option<Instance> {
        if pour.span.1 - pour.span.0 < 0.5 {
            return None;
        }
        let ripple = if pour.front {
            amplitude(self.radius) * 3.5 + 1.0
        } else {
            0.0
        };
        let c = self.centre();
        Some(Instance {
            quad: self.quad((pour.span.0, pour.span.1 + ripple), pour.tint.glow),
            shape: [c.0, c.1, self.radius, first as f32],
            color: pour.tint.top,
            color2: pour.tint.bottom,
            params: [
                kind::TUBE_LIQUID,
                pour.tint.glow,
                0.0,
                self.points.len() as f32,
            ],
            uv: [pour.span.0, pour.span.1, pour.jumped, 0.0],
            strain: Instance::strain_of(strain, pour.front),
        })
    }
}

/// The S from note `a` to note `b` (each from, to across and how far down;
/// `b` right after `a`) of a tube `radius` thick: where they meet, and how
/// far it reaches each way. Wider the further it goes, at most [`S_SHARE`]
/// of either note.
fn ess(a: (f32, f32, f32), b: (f32, f32, f32), radius: f32) -> (f32, f32) {
    let wide = (0.5 * (b.2 - a.2).abs() + 0.8 * radius)
        .min(S_SHARE * (a.1 - a.0))
        .min(S_SHARE * (b.1 - b.0));
    ((a.1 + b.0) / 2.0, wide.max(0.5))
}

/// How far down the middle of a tube through `notes` is at `x`: each note's
/// own height along it, an S (`smoothstep`) from one to the next round
/// their meeting ([`ess`]).
fn y_at(notes: &[(f32, f32, f32)], x: f32, radius: f32) -> f32 {
    let Some(&(_, _, mut y)) = notes.first() else {
        return 0.0;
    };
    for pair in notes.windows(2) {
        let (at, wide) = ess(pair[0], pair[1], radius);
        if x >= at - wide {
            y = pair[0].2 + (pair[1].2 - pair[0].2) * ease((x - (at - wide)) / (2.0 * wide));
        }
    }
    y
}

/// The nearest point to `p` of the middle line `points` (sorted across),
/// looked for within `reach` across: how far it is, where along the tube
/// `p` is (its place across, past the ends along the end pieces), and how
/// far it is to the side (positive below the line as it runs); a far way
/// off where nothing is within reach.
pub fn nearest(points: &[(f32, f32)], p: (f32, f32), reach: f32) -> (f32, f32, f32) {
    let n = points.len();
    let mut best = (1e9f32, p.0, 1e9f32);
    if n < 2 {
        return best;
    }
    // The first point not left of the reach, and the piece before it.
    let start = points.partition_point(|q| q.0 < p.0 - reach).max(1) - 1;
    let mut i = start;
    while i + 1 < n && points[i].0 <= p.0 + reach {
        let (a, b) = (points[i], points[i + 1]);
        let (ba, pa) = ((b.0 - a.0, b.1 - a.1), (p.0 - a.0, p.1 - a.1));
        let h = (pa.0 * ba.0 + pa.1 * ba.1) / (ba.0 * ba.0 + ba.1 * ba.1).max(1e-6);
        let hc = h.clamp(0.0, 1.0);
        let (dx, dy) = (pa.0 - ba.0 * hc, pa.1 - ba.1 * hc);
        let d = (dx * dx + dy * dy).sqrt();
        if d < best.0 {
            // Along: within the piece, or past an end along an end piece.
            let lo = if i == 0 { f32::MIN } else { 0.0 };
            let hi = if i + 2 == n { f32::MAX } else { 1.0 };
            let along = a.0 + ba.0 * h.clamp(lo, hi);
            let side = if ba.0 * pa.1 - ba.1 * pa.0 < 0.0 {
                -d
            } else {
                d
            };
            best = (d, along, side);
        }
        i += 1;
    }
    best
}

/// [`nearest`] in WGSL (`tube_near`, over the points `path_point` gives),
/// for the stage's shader and its check (`tube_near.wgsl`).
pub const NEAR_WGSL: &str = include_str!("tube_near.wgsl");

/// What the stage's shader takes of the tubes (it draws them itself, in
/// `stage.wgsl`): how much of a bulge the middle line takes, and
/// [`NEAR_WGSL`].
pub fn wgsl() -> String {
    format!("const PATH_GROW: f32 = {PATH_GROW:?};\n{NEAR_WGSL}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A word on one pitch is a straight capsule; on several, its middle
    /// line holds each note's pitch along it and goes smoothly between
    /// them, never past either, through round ends its radius in.
    #[test]
    fn a_tube_runs_through_its_notes_pitches() {
        let flat = Snake::through(
            &[(100.0, 200.0, 50.0), (200.0, 300.0, 50.0)],
            (100.0, 300.0),
            20.0,
        );
        assert_eq!(flat.points, vec![(120.0, 50.0), (280.0, 50.0)]);
        let notes = [
            (100.0, 200.0, 50.0),
            (200.0, 260.0, 10.0),
            (260.0, 400.0, 70.0),
        ];
        let s = Snake::through(&notes, (100.0, 400.0), 20.0);
        assert_eq!(s.points.first().unwrap().0, 120.0);
        assert_eq!(s.points.last().unwrap().0, 380.0);
        assert!(
            s.points.windows(2).all(|w| w[0].0 < w[1].0),
            "sorted across"
        );
        assert_eq!(s.y_at(150.0), 50.0);
        assert_eq!(s.y_at(230.0), 10.0);
        assert_eq!(s.y_at(350.0), 70.0);
        // Between: smooth (no step bigger than its S allows), within.
        for k in 0..300 {
            let x = 100.0 + k as f32;
            let (a, b) = (s.y_at(x), s.y_at(x + 1.0));
            assert!((10.0..=70.0).contains(&a), "{x}: {a}");
            assert!((a - b).abs() < 3.0, "{x}: {a} {b}");
        }
        // Too short for its ends: a disc.
        let dot = Snake::through(&[(0.0, 30.0, 5.0)], (0.0, 30.0), 20.0);
        assert_eq!(dot.points, vec![(15.0, 5.0), (15.0, 5.0)]);
        // Its text where it is flat: on the long last note, not over the
        // bends.
        let x = s.flattest(60.0, (110.0, 390.0));
        assert!(x >= 284.0 && x + 60.0 <= 390.0, "{x}");
        assert_eq!(s.y_at(x), s.y_at(x + 60.0));
        // On one pitch, in the middle.
        assert_eq!(flat.flattest(60.0, (110.0, 290.0)), 170.0);
    }

    /// The nearest point: its distance, where along, which side; past the
    /// ends along the end pieces.
    #[test]
    fn the_nearest_point_tells_how_far_where_along_and_which_side() {
        let pts = [(0.0, 0.0), (10.0, 0.0), (20.0, 10.0), (30.0, 10.0)];
        let (d, u, v) = nearest(&pts, (5.0, 3.0), 50.0);
        assert!((d - 3.0).abs() < 1e-5 && (u - 5.0).abs() < 1e-5 && v > 0.0);
        let (d, u, v) = nearest(&pts, (25.0, 6.0), 50.0);
        assert!((d - 4.0).abs() < 1e-5 && (u - 25.0).abs() < 1e-5 && v < 0.0);
        // Past the left end: along it, its distance from the end.
        let (d, u, _) = nearest(&pts, (-4.0, 3.0), 50.0);
        assert!((d - 5.0).abs() < 1e-5 && (u + 4.0).abs() < 1e-5);
        // Within the reach only.
        assert!(nearest(&pts, (100.0, 0.0), 5.0).0 > 1e8);
    }
}
