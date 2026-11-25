//! A pen on paper, for the diploma's sounds ([`super::diploma`]): a nib
//! scratching ([`scratch`]: noise through a band that rises with the pen's
//! speed, a moving formant, a little grit above it and the fibres catching
//! now and then, all as loud as the pen is fast; still or lifted, it is
//! silent), a hand writing words ([`handwriting`]: a letter after another,
//! the pen lifted between words), a tap of the nib ([`tap`]), and the short
//! bursts of noise the paper's sounds are made of ([`burst`]).
//!
//! What is written is not known when the sounds are made (they are made
//! once, for every result), so a hand writes a sample of what goes there
//! ("40 of 45", "17 in a row", [`DATE_SAMPLE`]; a grade is known, its
//! verdict's): its words, as the sample's spaces have them, are where the
//! pen lifts.

use std::f64::consts::PI;

use super::sounds::env;
use super::synth::{scatter, white, Biquad, Buf, Noise, Partial, Tone};

/// A short burst of noise through a band from `f.0` to `f.1` Hz (`q` its
/// sharpness), rising in `attack` seconds and dying away over `decay`,
/// `length` long, at `amp`.
pub fn burst(f: (f64, f64), q: f64, (attack, decay, length): (f64, f64, f64), amp: f64) -> Noise {
    Noise {
        from: f.0,
        to: f.1,
        q,
        envelope: env(attack, 0.0, decay, length),
        amp,
        length,
        seed: (f.0 * 7.0 + f.1 * 13.0 + length * 1e4) as u64 | 1,
    }
}

/// A pen scratching over paper from `start` for `length` seconds at `pan`,
/// as fast as `speed(t)` says (seconds from its start; 0 still or lifted,
/// 1 its fastest), its paper's grain from `seed`: noise through a band
/// whose middle rises with the speed, a little grit above it, the fibres
/// catching now and then (the more often the faster), all as loud as the
/// pen is fast.
pub fn scratch(
    b: &mut Buf,
    (start, length): (f64, f64),
    pan: f64,
    seed: u64,
    speed: impl Fn(f64) -> f64,
) {
    let rate = b.rate;
    let mut seed = seed | 1;
    let mut body = Biquad::band(2_000.0, 1.4, rate);
    let mut grit = Biquad::band(6_500.0, 1.0, rate);
    // The speed eased over 5 ms (no clicks where the pen goes down or
    // comes off), the paper's fibres a slow noise, a catch dying in 1 ms.
    let ease = 1.0 - (-1.0 / (0.005 * rate)).exp();
    let fibres = 1.0 - (-2.0 * PI * 70.0 / rate).exp();
    let let_go = (-1.0 / (0.001 * rate)).exp();
    let (mut v, mut fibre, mut catch) = (0.0f64, 0.0f64, 0.0f64);
    let mut k = 0usize;
    b.add(start, length, pan, |t| {
        v += ease * (speed(t).clamp(0.0, 1.0) - v);
        fibre += fibres * (white(&mut seed) - fibre);
        if k.is_multiple_of(32) {
            let f = 1_200.0 + 3_600.0 * v + 2_000.0 * fibre;
            body = body.retuned(f.max(400.0), 1.4, rate);
        }
        k += 1;
        if scatter(&mut seed) < v * 40.0 / rate {
            catch = 1.0;
        }
        catch *= let_go;
        let x = white(&mut seed) * (1.0 + 5.0 * catch);
        let grain = (0.55 + 6.0 * fibre.abs()).min(1.6);
        v.powf(0.8) * grain * (body.run(x) + 0.3 * grit.run(x))
    });
}

/// A hand writing `sample` over `length` seconds: a letter after another
/// (each two strokes, a little faster or slower, by `seed`), the pen
/// lifted for a moment where the sample's words part, and off at the end.
pub(super) struct Hand {
    letters: Vec<f64>,
    lifts: Vec<f64>,
    length: f64,
}

/// A date as the diploma writes one ("25 November 2026"), to write in for
/// whatever day it is: its words are a day's, a month's and a year's, as
/// long as most.
pub const DATE_SAMPLE: &str = "25 November 2026";

/// How long the pen is off the paper between two words.
const LIFT: f64 = 0.045;

impl Hand {
    pub(super) fn new(length: f64, sample: &str, seed: u64) -> Hand {
        let mut seed = seed | 1;
        let mut letters = vec![0.0];
        let mut at = 0.0;
        while at < length {
            at += 0.055 + 0.035 * scatter(&mut seed);
            letters.push(at);
        }
        let chars = sample.chars().count().max(1) as f64;
        let lifts = sample
            .chars()
            .enumerate()
            .filter(|(_, c)| *c == ' ')
            .map(|(k, _)| (k as f64 + 0.5) / chars * length)
            .collect();
        Hand {
            letters,
            lifts,
            length,
        }
    }

    /// How fast the pen goes at `t`, 0 to 1.
    pub(super) fn speed(&self, t: f64) -> f64 {
        let lifted = self.lifts.iter().any(|l| (t - l).abs() < LIFT / 2.0);
        if !(0.0..self.length).contains(&t) || lifted {
            return 0.0;
        }
        let i = self.letters.partition_point(|&m| m <= t).max(1) - 1;
        let (from, to) = (self.letters[i], self.letters[i + 1]);
        let u = (t - from) / (to - from);
        0.3 + 0.7 * (2.0 * PI * u).sin().abs()
    }
}

/// `sample` written in by hand from `start` over `length` seconds, at
/// `pan`, its paper's grain from `seed`.
pub fn handwriting(
    b: &mut Buf,
    (start, length): (f64, f64),
    sample: &str,
    (pan, seed): (f64, u64),
) {
    let hand = Hand::new(length, sample, seed);
    scratch(b, (start, length + 0.02), pan, seed, |t| hand.speed(t));
}

/// A soft tap of the pen at `at`: a click of the nib and a little knock
/// of the desk under the paper.
pub fn tap(b: &mut Buf, at: f64, amp: f64) {
    b.noise(
        at,
        0.0,
        &burst((2_600.0, 1_800.0), 2.0, (0.0003, 0.002, 0.01), amp),
    );
    let wood = [Partial::new(1.0, 1.0, 0.02)];
    let knock = Tone::new(620.0, env(0.0008, 0.0, 0.012, 0.05), &wood)
        .glide(0.85, 0.02)
        .amp(0.5 * amp);
    b.tone(at, 0.0, &knock);
}

/// A sound's loudness in windows of 10 ms from `from` to `to` seconds (its
/// left side).
#[cfg(test)]
pub(super) fn loudness(b: &Buf, from: f64, to: f64) -> Vec<f64> {
    let w = (0.01 * b.rate) as usize;
    let (a, z) = ((from * b.rate) as usize, (to * b.rate) as usize);
    b.l[a..z.min(b.l.len())]
        .chunks(w)
        .map(|c| (c.iter().map(|v| v * v).sum::<f64>() / c.len() as f64).sqrt())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pen lifts between the sample's words: silent there, writing
    /// either side, and off at the end.
    #[test]
    fn a_hand_lifts_between_words() {
        // "40 of 45": words parted after its 2nd and 5th characters.
        let hand = Hand::new(0.8, "40 of 45", 7);
        assert_eq!(hand.speed(0.25), 0.0);
        assert_eq!(hand.speed(0.55), 0.0);
        assert!(hand.speed(0.1) > 0.0 && hand.speed(0.4) > 0.0 && hand.speed(0.7) > 0.0);
        assert_eq!(hand.speed(0.81), 0.0);
        let mut b = Buf::new(48_000, 0.9);
        handwriting(&mut b, (0.0, 0.8), "40 of 45", (0.0, 7));
        let heard = loudness(&b, 0.0, 0.8);
        let most = heard.iter().copied().fold(0.0, f64::max);
        // The window round the first lift (0.25 s), quiet against the rest.
        assert!(heard[25] < 0.25 * most, "{} of {most}", heard[25]);
    }
}
