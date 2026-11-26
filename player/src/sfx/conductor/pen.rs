//! A pen on paper, for the diploma's sheet ([`super::sheet`]): a nib
//! scratching ([`scratch`]: grains of noise through a band that rises with
//! the pen's speed, the paper's fibres scattering it, the fibres catching
//! now and then, all as loud as the pen is fast; still or lifted, it is
//! silent), a hand writing words ([`handwriting`]: a letter after another,
//! the pen lifted between words), the signature's very stroke
//! ([`stroke_speed`]) and a tap of the nib ([`tap`]).
//!
//! What is written is not known when its sound is planned (a value lands
//! as it is drawn), so a hand writes a sample of what goes there ("40 of
//! 45", "17 in a row", [`DATE_SAMPLE`]; a grade is known, its verdict's):
//! its words, as the sample's spaces have them, are where the pen lifts.
//! Nothing here allocates.

use std::f64::consts::TAU;

use super::palette::Atom;
use super::sheet::{midi, Paper};
use crate::core::karaoke::result::signature::signature_point;
use crate::core::karaoke::result::{signed, SIGNING};

/// A grain of the pen's scratch, seconds: a note on the nib each
/// ([`Atom::Nib`]).
pub(super) const GRAIN: f64 = 0.012;

/// A date as the diploma writes one ("25 November 2026"), to write in for
/// whatever day it is: its words are a day's, a month's and a year's, as
/// long as most.
pub(super) const DATE_SAMPLE: &str = "25 November 2026";

/// How long the pen is off the paper between two words.
const LIFT: f64 = 0.045;

/// The steps the signature's stroke is followed in, seconds.
const STEP: f64 = 0.001;

/// A pen scratching over paper from `start` for `length` seconds at `pan`,
/// as fast as `speed(t)` says (seconds from its start, never going back; 0
/// still or lifted, 1 its fastest): a grain at a time, its band higher and
/// its grain louder the faster the pen goes, the fibres catching now and
/// then (the more often the faster).
pub(super) fn scratch(
    p: &mut Paper,
    (start, length): (f64, f64),
    pan: f32,
    mut speed: impl FnMut(f64) -> f64,
) {
    let grains = (length / GRAIN).ceil() as u32;
    for k in 0..grains {
        let t = f64::from(k) * GRAIN;
        let v = speed(t + 0.5 * GRAIN).clamp(0.0, 1.0);
        if v < 0.02 {
            continue;
        }
        let fibre = f64::from(p.unit());
        let band = 1_500.0 + 4_500.0 * v + 1_000.0 * (fibre - 0.5);
        let loud = v.powf(0.8) * (0.7 + 0.6 * fibre);
        p.play(start + t, Atom::Nib, midi(band), loud as f32, pan, GRAIN);
        if f64::from(p.unit()) < v * 40.0 * GRAIN {
            let at = start + t + GRAIN * f64::from(p.unit());
            let f = midi(2_500.0 + 2_500.0 * f64::from(p.unit()));
            p.play(at, Atom::Snap, f, 0.5 * v as f32, pan, 0.01);
        }
    }
}

/// A hand writing a sample of text over its length: a letter after
/// another (each two strokes, a little faster or slower, by its seed), the
/// pen lifted for a moment where the sample's words part, and off at the
/// end. Asked from its start on, never going back ([`Hand::speed`]).
struct Hand<'a> {
    sample: &'a str,
    length: f64,
    seed: u32,
    /// The letter being written: its start and its end.
    letter: (f64, f64),
}

impl<'a> Hand<'a> {
    /// A hand writing `sample` over `length` seconds, its letters' pace
    /// from `seed`.
    fn new(length: f64, sample: &'a str, seed: u32) -> Hand<'a> {
        let mut hand = Hand {
            sample,
            length,
            seed: seed | 1,
            letter: (0.0, 0.0),
        };
        hand.letter.1 = hand.pace();
        hand
    }

    /// How long the next letter takes.
    fn pace(&mut self) -> f64 {
        let x = jodel_dsp::xorshift32(&mut self.seed);
        0.055 + 0.035 * f64::from(x >> 8) / f64::from(1u32 << 24)
    }

    /// Whether the pen is lifted at `t`, between two words.
    fn lifted(&self, t: f64) -> bool {
        let chars = self.sample.chars().count().max(1) as f64;
        (self.sample.chars().enumerate())
            .filter(|&(_, c)| c == ' ')
            .any(|(k, _)| ((k as f64 + 0.5) / chars * self.length - t).abs() < LIFT / 2.0)
    }

    /// How fast the pen goes at `t`, 0 to 1 (`t` from its start, never
    /// going back).
    fn speed(&mut self, t: f64) -> f64 {
        if !(0.0..self.length).contains(&t) || self.lifted(t) {
            return 0.0;
        }
        while self.letter.1 <= t {
            let next = self.pace();
            self.letter = (self.letter.1, self.letter.1 + next);
        }
        let (from, to) = self.letter;
        let u = (t - from) / (to - from);
        0.3 + 0.7 * (TAU * u).sin().abs()
    }
}

/// `sample` written in by hand from `start` over `length` seconds, the
/// pen at `velocity` (1: full), its letters' pace from `seed`.
pub(super) fn handwriting(
    p: &mut Paper,
    (start, length): (f64, f64),
    sample: &str,
    (velocity, seed): (f64, u32),
) {
    let mut hand = Hand::new(length, sample, seed);
    scratch(p, (start, length + 0.02), 0.0, |t| velocity * hand.speed(t));
}

/// A soft tap of the pen at `t` seconds, at `velocity`: a click of the nib
/// and a little knock of the desk under the paper.
pub(super) fn tap(p: &mut Paper, t: f64, velocity: f32) {
    p.play(t, Atom::Snap, midi(1_800.0), velocity, 0.0, 0.01);
    p.play(t, Atom::Knock, midi(527.0), 0.5 * velocity, 0.0, 0.05);
}

/// How fast the signature's pen goes along its stroke
/// ([`signature_point`]) `a` seconds after the end, as the stage writes it
/// ([`signed`]), its share of a second over the next step.
pub(super) fn stroke_speed(a: f64) -> f64 {
    let at = |a: f64| {
        let (x, y) = signature_point(signed(a) as f32);
        (f64::from(x), f64::from(y))
    };
    let (p, q) = (at(a), at(a + STEP));
    (q.0 - p.0).hypot(q.1 - p.1) / STEP
}

/// The signature's pen at its fastest ([`stroke_speed`]'s most).
pub(super) fn fastest() -> f64 {
    let steps = (SIGNING / STEP).ceil() as u32;
    (0..steps)
        .map(|k| stroke_speed(f64::from(k) * STEP))
        .fold(1e-9, f64::max)
}

/// How fast the signature's pen goes over the grain from `a` seconds, 0 to
/// 1 (its fastest: `most`): the mean of its steps there.
pub(super) fn signature_speed(a: f64, most: f64) -> f64 {
    let steps = (GRAIN / STEP).round() as u32;
    let sum: f64 = (0..steps)
        .map(|k| stroke_speed(a + f64::from(k) * STEP))
        .sum();
    sum / f64::from(steps) / most
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pen lifts between the sample's words: silent there, writing
    /// either side, and off at the end.
    #[test]
    fn a_hand_lifts_between_words() {
        // "40 of 45": words parted after its 2nd and 5th characters.
        let mut hand = Hand::new(0.8, "40 of 45", 7);
        assert!(hand.speed(0.1) > 0.0);
        assert_eq!(hand.speed(0.25), 0.0);
        assert!(hand.speed(0.4) > 0.0);
        assert_eq!(hand.speed(0.55), 0.0);
        assert!(hand.speed(0.7) > 0.0);
        assert_eq!(hand.speed(0.81), 0.0);
    }
}
