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

/// The signature's loops at `v` (0 to 1) along them: five and a half,
/// the first tall, leaning right, on a line.
fn loops(v: f32) -> (f32, f32) {
    let w = TAU * 5.5 * v;
    let c = 0.07 + 0.24 * (-v * 6.0).exp() + 0.04 * (v * 13.0 + 1.0).sin();
    let y = 0.1 - 2.0 * c * (0.5 - 0.5 * w.cos());
    (-0.92 + 1.7 * v - 0.085 * w.sin() - 0.35 * (y - 0.1), y)
}

/// The signature's pen at `u` (0 to 1) along it (its width about 2, its
/// middle at 0, down positive): its loops, then a swash back under them.
pub fn signature_point(u: f32) -> (f32, f32) {
    if u <= LOOPS {
        return loops(u / LOOPS);
    }
    let q = (u - LOOPS) / (1.0 - LOOPS);
    let (a, b, c) = (loops(1.0), (1.1, 0.34), (-0.8, 0.22));
    let (p, r) = ((1.0 - q) * (1.0 - q), q * q);
    let m = 2.0 * q * (1.0 - q);
    (p * a.0 + m * b.0 + r * c.0, p * a.1 + m * b.1 + r * c.1)
}

/// The signature's point `i` of [`PEN`] + 1 along it: what it is drawn
/// through, on the processor and (as a table) on the GPU.
pub fn pen_point(i: u32) -> (f32, f32) {
    signature_point(i as f32 / PEN as f32)
}
