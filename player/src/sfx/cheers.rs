//! The result screen's verdicts, made here at the desk's rate like its
//! others ([`super::sounds`]):
//!
//! - [`Sound::Medal`]: the medal's wax seal pressed down: a thump, the
//!   wax's soft squelch and a crack, then its stinger: Bronze a warm
//!   plucked chord, Silver bells ringing up to a chord, Gold a
//!   "ta-da-daa" in brass over bells and a cascade of glitter.
//! - [`Sound::NewBest`]: a new best: its rubber stamp slapped on
//!   ([`super::diploma::rubber_stamp`]), then a fanfare, four brass notes
//!   rising to a held chord, a roll of noise under them and a shimmer over
//!   it.
//!
//! [`Sound::Medal`]: super::Sound::Medal
//! [`Sound::NewBest`]: super::Sound::NewBest

use super::sounds::{brass, env, penta, pluck, BELL, GLASS};
use super::synth::{scatter, Buf, Noise, Partial, Tone};
use crate::core::singing::{Medal, STAMP_DOWN};

/// The medals, in the bank's order.
pub const MEDALS: [Medal; 3] = [Medal::Bronze, Medal::Silver, Medal::Gold];

/// The wax seal pressed down under a medal: a thump sliding down, the
/// soft squelch of the wax giving under it, and a crack of noise.
fn stamp(b: &mut Buf) {
    let sine = [Partial::new(1.0, 1.0, f64::INFINITY)];
    let thump = Tone::new(140.0, env(0.001, 0.01, 0.06, 0.25), &sine).glide(0.4, 0.05);
    b.tone(0.0, 0.0, &thump);
    let noise = |from, to, q, decay, amp, seed| Noise {
        from,
        to,
        q,
        envelope: env(0.0005, 0.0, decay, 4.0 * decay),
        amp,
        length: 4.0 * decay,
        seed,
    };
    b.noise(0.0, 0.0, &noise(2_500.0, 600.0, 0.8, 0.03, 0.6, 0x57A3));
    b.noise(0.004, 0.0, &noise(520.0, 230.0, 1.4, 0.05, 1.1, 0x3A1));
}

/// `medal` stamped down, and its stinger.
pub fn medal(rate: u32, medal: Medal) -> Buf {
    let mut b = Buf::new(rate, 2.2);
    stamp(&mut b);
    super::diploma::medal_grade(&mut b, medal);
    let at = 0.12;
    match medal {
        Medal::Bronze => {
            for (i, s) in [0, 2, 4, 5].into_iter().enumerate() {
                let pl = Tone::new(penta(s), env(0.002, 0.03, 0.35, 1.0), &pluck(0.5)).amp(0.5);
                b.tone(at + 0.03 * i as f64, -0.3 + 0.2 * i as f64, &pl);
            }
        }
        Medal::Silver => {
            for (i, s) in [3, 5, 7].into_iter().enumerate() {
                let bell = Tone::new(penta(s), env(0.0015, 0.0, 0.4, 1.0), &BELL).amp(0.5);
                b.tone(at + 0.09 * i as f64, -0.3 + 0.3 * i as f64, &bell);
            }
            for (i, s) in [5, 7, 8, 10].into_iter().enumerate() {
                let pl = Tone::new(penta(s), env(0.002, 0.05, 0.4, 1.2), &pluck(0.6)).amp(0.35);
                b.tone(at + 0.3, -0.45 + 0.3 * i as f64, &pl);
            }
        }
        Medal::Gold => {
            for (i, f) in [392.0, 493.9, 587.3].into_iter().enumerate() {
                b.tone(at, -0.3 + 0.3 * i as f64, &brass(f, 0.04, 0.13).amp(0.45));
            }
            for (i, f) in [523.3, 659.3, 784.0, 1046.5].into_iter().enumerate() {
                let held = brass(f, 0.45, 1.3).amp(0.42);
                b.tone(at + 0.17, -0.45 + 0.3 * i as f64, &held);
                let bell = Tone::new(2.0 * f, env(0.0015, 0.0, 0.4, 1.0), &BELL).amp(0.2);
                b.tone(at + 0.17, 0.45 - 0.3 * i as f64, &bell);
            }
            let mut seed = 0x601D_u64;
            for g in 0..10 {
                let t = at + 0.25 + 0.045 * f64::from(g);
                let grain = Tone::new(penta(9 + g), env(0.001, 0.0, 0.1, 0.3), &GLASS).amp(0.3);
                b.tone(t, 1.6 * scatter(&mut seed) - 0.8, &grain);
            }
        }
    }
    b.room(0.3, 1.0);
    b
}

/// A new best: its stamp slapped on, then its fanfare.
pub fn new_best(rate: u32) -> Buf {
    let mut b = Buf::new(rate, 2.65);
    super::diploma::rubber_stamp(&mut b, 0.0);
    let mut fanfare = Buf::new(rate, 2.4);
    fanfare_into(&mut fanfare);
    b.mix(STAMP_DOWN + 0.12, &fanfare, 0.8);
    b
}

/// A new best's fanfare, into `b`.
fn fanfare_into(b: &mut Buf) {
    // A roll of noise under it, swelling.
    b.noise(
        0.0,
        0.0,
        &Noise {
            from: 300.0,
            to: 1_200.0,
            q: 0.9,
            envelope: env(0.35, 0.05, 0.08, 0.5),
            amp: 0.25,
            length: 0.5,
            seed: 0xBE57,
        },
    );
    // "Da da da daaa": G, C, E, then a C major chord, held.
    for (i, f) in [392.0, 523.3, 659.3].into_iter().enumerate() {
        let t = 0.12 * i as f64;
        b.tone(t, 0.0, &brass(f, 0.03, 0.11).amp(0.5));
        b.tone(t, 0.0, &brass(f / 2.0, 0.03, 0.11).amp(0.25));
    }
    for (i, f) in [523.3, 659.3, 784.0, 1046.5, 1318.5]
        .into_iter()
        .enumerate()
    {
        b.tone(0.36, -0.5 + 0.25 * i as f64, &brass(f, 0.6, 1.5).amp(0.36));
    }
    let mut seed = 0xF4F4_u64;
    for g in 0..12 {
        let t = 0.4 + 0.06 * f64::from(g);
        let grain = Tone::new(penta(10 + g % 6), env(0.001, 0.0, 0.1, 0.3), &GLASS).amp(0.25);
        b.tone(t, 1.6 * scatter(&mut seed) - 0.8, &grain);
    }
    b.room(0.35, 1.2);
}
