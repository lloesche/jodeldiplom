//! The examiner's signature on the result's diploma, as a stroke: five
//! and a half loops leaning right, then a swash back under them
//! ([`signature_point`]), drawn through [`PEN`] straight steps
//! ([`pen_point`]). The stage draws it (on the GPU, its points as a table)
//! and the sounds follow it (the pen's scratch at its pace), from this one
//! stroke.

use std::f32::consts::TAU;

/// How far the signature's loops go along it (the rest: its swash).
const LOOPS: f32 = 0.78;
/// How many straight steps the signature is drawn in.
pub const PEN: u32 = 120;

/// The loops' turns (five and a half), the line they sit on, where they
/// start across and how far they go, how far each turn sways across, and
/// how far they lean right with their height.
const TURNS: f32 = 5.5;
const LINE: f32 = 0.1;
const START: f32 = -0.92;
const ALONG: f32 = 1.7;
const SWAY: f32 = 0.085;
const LEAN: f32 = 0.35;
/// The loops' height: a base, a first tall loop falling away (its height
/// and how fast it falls), and a ripple (its height and how fast).
const BASE: f32 = 0.07;
const TALL: (f32, f32) = (0.24, 6.0);
const RIPPLE: (f32, f32) = (0.04, 13.0);

/// The loops' height at `v`, and how it goes (its slope).
fn height(v: f32) -> (f32, f32) {
    let (fall, ripple) = ((-v * TALL.1).exp(), v * RIPPLE.1 + 1.0);
    let c = BASE + TALL.0 * fall + RIPPLE.0 * ripple.sin();
    let dc = -TALL.0 * TALL.1 * fall + RIPPLE.0 * RIPPLE.1 * ripple.cos();
    (c, dc)
}

/// The signature's loops at `v` (0 to 1) along them: five and a half,
/// the first tall, leaning right, on a line.
fn loops(v: f32) -> (f32, f32) {
    let w = TAU * TURNS * v;
    let (c, _) = height(v);
    let y = LINE - 2.0 * c * (0.5 - 0.5 * w.cos());
    (START + ALONG * v - SWAY * w.sin() - LEAN * (y - LINE), y)
}

/// How the loops go at `v` ([`loops`]'s slope, a whole `v` of 1).
fn loops_slope(v: f32) -> (f32, f32) {
    let dw = TAU * TURNS;
    let w = dw * v;
    let (c, dc) = height(v);
    // y = LINE - c (1 - cos w).
    let dy = -dc * (1.0 - w.cos()) - c * w.sin() * dw;
    (ALONG - SWAY * w.cos() * dw - LEAN * dy, dy)
}

/// The swash's ends and the point it bends towards.
const SWASH: [(f32, f32); 2] = [(1.1, 0.34), (-0.8, 0.22)];

/// The signature's pen at `u` (0 to 1) along it (its width about 2, its
/// middle at 0, down positive): its loops, then a swash back under them.
pub fn signature_point(u: f32) -> (f32, f32) {
    if u <= LOOPS {
        return loops(u / LOOPS);
    }
    let q = (u - LOOPS) / (1.0 - LOOPS);
    let (a, [b, c]) = (loops(1.0), SWASH);
    let (p, r) = ((1.0 - q) * (1.0 - q), q * q);
    let m = 2.0 * q * (1.0 - q);
    (p * a.0 + m * b.0 + r * c.0, p * a.1 + m * b.1 + r * c.1)
}

/// How fast the signature's pen goes at `u` along it: the length of
/// [`signature_point`]'s slope (of its width for the whole of `u`).
pub fn signature_pace(u: f32) -> f32 {
    let (dx, dy) = if u <= LOOPS {
        let (dx, dy) = loops_slope(u / LOOPS);
        (dx / LOOPS, dy / LOOPS)
    } else {
        let q = (u - LOOPS) / (1.0 - LOOPS);
        let (a, [b, c]) = (loops(1.0), SWASH);
        // The swash's quadratic, its slope by q, then by u.
        let slope = |a: f32, b: f32, c: f32| {
            (2.0 * (1.0 - q) * (b - a) + 2.0 * q * (c - b)) / (1.0 - LOOPS)
        };
        (slope(a.0, b.0, c.0), slope(a.1, b.1, c.1))
    };
    dx.hypot(dy)
}

/// The signature's point `i` of [`PEN`] + 1 along it: what it is drawn
/// through, on the processor and (as a table) on the GPU.
pub fn pen_point(i: u32) -> (f32, f32) {
    signature_point(i as f32 / PEN as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pen's pace along the stroke is the stroke's own: the distance
    /// between two points close by, over how far apart they are.
    #[test]
    fn the_pace_is_the_stroke_s() {
        let h = 1e-3f64;
        for k in 1..200 {
            let u = f64::from(k) / 200.0;
            if (u - f64::from(LOOPS)).abs() < h {
                // The loops' end turns into the swash: no slope there.
                continue;
            }
            let at = |u: f64| {
                let (x, y) = signature_point(u as f32);
                (f64::from(x), f64::from(y))
            };
            let (p, q) = (at(u - h / 2.0), at(u + h / 2.0));
            let stepped = (q.0 - p.0).hypot(q.1 - p.1) / h;
            let pace = f64::from(signature_pace(u as f32));
            assert!(
                (pace - stepped).abs() <= 0.01 * stepped.max(1.0),
                "{u}: {pace} against {stepped}"
            );
        }
    }
}
