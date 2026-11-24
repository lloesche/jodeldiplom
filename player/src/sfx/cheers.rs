//! The result screen's sounds, made here at the desk's rate like the
//! game's others ([`super::sounds`]):
//!
//! - [`Sound::Tick`]: the points counting up: a short bright blip, a step
//!   of the scale higher as the count goes, a click of noise on it.
//! - [`Sound::Gem`]: a phrase's gem popping in: a glassy plink, higher and
//!   richer by its tier (none a soft tink, Good a chime, Great two bells,
//!   Amazing a bright pair with a shimmer, Diva a sparkling run).
//! - [`Sound::Medal`]: the medal stamped down: a thump and a crack of
//!   noise, then its stinger: Bronze a warm plucked chord, Silver bells
//!   ringing up to a chord, Gold a "ta-da-daa" in brass over bells and a
//!   cascade of glitter.
//! - [`Sound::NewBest`]: a new best: a fanfare, four brass notes rising to
//!   a held chord, a roll of noise under them and a shimmer over it.
//!
//! [`Sound::Tick`]: super::Sound::Tick
//! [`Sound::Gem`]: super::Sound::Gem
//! [`Sound::Medal`]: super::Sound::Medal
//! [`Sound::NewBest`]: super::Sound::NewBest

use super::sounds::{brass, env, penta, pluck, BELL, GLASS};
use super::synth::{scatter, Buf, Noise, Partial, Tone};
use crate::core::shine::Tier;
use crate::core::singing::Medal;

/// The steps of the scale the count-up's ticks climb.
pub const TICK_STEPS: usize = 8;
/// The medals, in the bank's order.
pub const MEDALS: [Medal; 3] = [Medal::Bronze, Medal::Silver, Medal::Gold];

/// A square-ish tone's partials, short.
fn blip() -> Vec<Partial> {
    [1.0, 3.0, 5.0]
        .map(|n: f64| Partial::new(n, 1.0 / n, 0.03 / n))
        .to_vec()
}

/// Tick `step` of the count-up.
pub fn tick(rate: u32, step: usize) -> Buf {
    let mut b = Buf::new(rate, 0.07);
    let f = penta(7 + step as i32);
    b.tone(
        0.0,
        0.0,
        &Tone::new(f, env(0.0005, 0.006, 0.012, 0.05), &blip()),
    );
    b.noise(
        0.0,
        0.0,
        &Noise {
            from: 6_000.0,
            to: 5_000.0,
            q: 1.5,
            envelope: env(0.0002, 0.0, 0.002, 0.006),
            amp: 0.5,
            length: 0.006,
            seed: 0x71C + step as u64,
        },
    );
    b
}

/// A gem of `tier` popping in.
pub fn gem(rate: u32, tier: Option<Tier>) -> Buf {
    let mut b = Buf::new(rate, 0.7);
    let plink = |b: &mut Buf, at: f64, s: i32, pan: f64, amp: f64| {
        let t = Tone::new(penta(s), env(0.001, 0.0, 0.12, 0.4), &GLASS).amp(amp);
        b.tone(at, pan, &t);
    };
    match tier {
        None => plink(&mut b, 0.0, 3, 0.0, 0.6),
        Some(Tier::Good) => {
            let chime = Tone::new(penta(6), env(0.0015, 0.0, 0.16, 0.45), &BELL);
            b.tone(0.0, 0.0, &chime);
        }
        Some(Tier::Great) => {
            for (i, s) in [7, 9].into_iter().enumerate() {
                let bell = Tone::new(penta(s), env(0.0015, 0.0, 0.15, 0.4), &BELL);
                b.tone(0.035 * i as f64, -0.2 + 0.4 * i as f64, &bell);
            }
        }
        Some(Tier::Amazing) => {
            plink(&mut b, 0.0, 9, -0.2, 1.0);
            plink(&mut b, 0.03, 11, 0.2, 0.8);
            let bell = Tone::new(penta(4), env(0.001, 0.0, 0.2, 0.5), &BELL).amp(0.4);
            b.tone(0.0, 0.0, &bell);
        }
        Some(Tier::Diva) => {
            let mut seed = 0xD1A_u64;
            for g in 0..5 {
                let pan = 1.2 * scatter(&mut seed) - 0.6;
                plink(
                    &mut b,
                    0.028 * f64::from(g),
                    10 + g,
                    pan,
                    1.0 - 0.12 * f64::from(g),
                );
            }
            let bell = Tone::new(penta(5), env(0.001, 0.0, 0.3, 0.6), &BELL).amp(0.5);
            b.tone(0.0, 0.0, &bell);
        }
    }
    b.room(0.25, 0.5);
    b
}

/// The stamp under a medal: a thump sliding down and a crack of noise.
fn stamp(b: &mut Buf) {
    let sine = [Partial::new(1.0, 1.0, f64::INFINITY)];
    let thump = Tone::new(140.0, env(0.001, 0.01, 0.06, 0.25), &sine).glide(0.4, 0.05);
    b.tone(0.0, 0.0, &thump);
    b.noise(
        0.0,
        0.0,
        &Noise {
            from: 2_500.0,
            to: 600.0,
            q: 0.8,
            envelope: env(0.0005, 0.0, 0.03, 0.12),
            amp: 0.9,
            length: 0.12,
            seed: 0x57A3,
        },
    );
}

/// `medal` stamped down, and its stinger.
pub fn medal(rate: u32, medal: Medal) -> Buf {
    let mut b = Buf::new(rate, 2.2);
    stamp(&mut b);
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

/// A new best's fanfare.
pub fn new_best(rate: u32) -> Buf {
    let mut b = Buf::new(rate, 2.4);
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
    b
}
