//! The tiers' own sounds, each a recipe a row of the tier table names
//! with the peak it is made at ([`crate::tiers::TIERS`], [`Recipe`]),
//! made at the desk's rate like the game's others ([`super::sounds`]):
//!
//! - A phrase's chime ([`super::Sound::Tier`]): a two-note chime
//!   ([`good`]), a bright arpeggio with glitter ([`great`]), a "ta-daa" in
//!   brass with a shimmer ([`amazing`]), a jackpot (a cascade of coins
//!   over bells, a whoosh, and a big chord: [`diva`]).
//! - Its gem popping in on the diploma ([`super::Sound::Gem`]): a glassy
//!   plink, higher and richer up the tiers (a chime, two bells, a bright
//!   pair with a shimmer, a sparkling run).

use super::sounds::{brass, env, penta, pluck, BELL, GLASS};
use super::synth::{scatter, Buf, Noise, Partial, Tone};

/// A sound of a tier's: how it is made at a rate (stereo), and the peak
/// it is set to, dBFS.
#[derive(Clone, Copy, Debug)]
pub struct Recipe {
    /// It, made at a rate.
    pub make: fn(u32) -> Buf,
    /// Its peak.
    pub peak_db: f64,
}

/// Good's chime: two bell notes, a step apart.
pub fn good(rate: u32) -> Buf {
    let mut b = Buf::new(rate, 0.8);
    for (i, s) in [3, 5].into_iter().enumerate() {
        let chime = Tone::new(penta(s), env(0.0015, 0.0, 0.3, 0.6), &BELL);
        b.tone(0.09 * i as f64, -0.2 + 0.4 * i as f64, &chime);
    }
    b.room(0.25, 0.7);
    b
}

/// Great's: a plucked arpeggio with bells an octave up, and glitter.
pub fn great(rate: u32) -> Buf {
    let mut b = Buf::new(rate, 1.1);
    for (i, s) in [0, 2, 3, 5].into_iter().enumerate() {
        let t = 0.06 * i as f64;
        let pan = -0.4 + 0.8 * i as f64 / 3.0;
        let pl = Tone::new(penta(s), env(0.002, 0.0, 0.25, 0.55), &pluck(0.6));
        b.tone(t, pan, &pl);
        let bell = Tone::new(2.0 * penta(s), env(0.0015, 0.0, 0.2, 0.4), &BELL).amp(0.3);
        b.tone(t, pan, &bell);
    }
    let mut seed = 0x6EA7_u64;
    for g in 0..5 {
        let at = 0.26 + 0.035 * f64::from(g);
        let f = penta(10 + g);
        let grain = Tone::new(f, env(0.001, 0.0, 0.08, 0.25), &GLASS).amp(0.35);
        b.tone(at, 1.6 * scatter(&mut seed) - 0.8, &grain);
    }
    b.room(0.3, 0.9);
    b
}

/// Amazing's: a "ta-daa" in brass, a shimmer going up over it.
pub fn amazing(rate: u32) -> Buf {
    let mut b = Buf::new(rate, 1.6);
    // "Ta": G major, short; "daa": C major, held.
    for (i, f) in [392.0, 493.9, 587.3].into_iter().enumerate() {
        b.tone(0.0, -0.3 + 0.3 * i as f64, &brass(f, 0.04, 0.14).amp(0.5));
    }
    for (i, f) in [523.3, 659.3, 784.0, 1046.5].into_iter().enumerate() {
        b.tone(
            0.16,
            -0.45 + 0.3 * i as f64,
            &brass(f, 0.35, 0.95).amp(0.45),
        );
    }
    // A shimmer going up over the held chord.
    let mut seed = 0xA3A3_u64;
    for g in 0..8 {
        let at = 0.2 + 0.05 * f64::from(g);
        let grain = Tone::new(penta(8 + g), env(0.001, 0.0, 0.1, 0.3), &GLASS).amp(0.25);
        b.tone(at, 1.6 * scatter(&mut seed) - 0.8, &grain);
    }
    b.noise(
        0.1,
        0.0,
        &Noise {
            from: 800.0,
            to: 7_000.0,
            q: 1.5,
            envelope: env(0.05, 0.02, 0.06, 0.2),
            amp: 0.12,
            length: 0.2,
            seed: 0xBEEF,
        },
    );
    b.room(0.3, 1.0);
    b
}

/// A coin: a quick blip on `f`, then a fourth up, ringing a little.
fn coin(b: &mut Buf, at: f64, f: f64, pan: f64, amp: f64) {
    let square: Vec<Partial> = [1.0, 3.0, 5.0, 7.0]
        .into_iter()
        .map(|n| Partial::new(n, 1.0 / n, 0.4 / n))
        .collect();
    b.tone(
        at,
        pan,
        &Tone::new(f, env(0.0015, 0.035, 0.01, 0.05), &square).amp(0.5 * amp),
    );
    b.tone(
        at + 0.05,
        pan,
        &Tone::new(f * 4.0 / 3.0, env(0.0015, 0.03, 0.09, 0.25), &square).amp(0.5 * amp),
    );
}

/// Diva's: a jackpot, a whoosh up into bells, a cascade of coins and a
/// big chord.
pub fn diva(rate: u32) -> Buf {
    let mut b = Buf::new(rate, 2.9);
    // A whoosh up into it.
    b.noise(
        0.0,
        0.0,
        &Noise {
            from: 500.0,
            to: 8_000.0,
            q: 1.2,
            envelope: env(0.2, 0.02, 0.06, 0.32),
            amp: 0.2,
            length: 0.32,
            seed: 0xD1FA,
        },
    );
    // Bells, one a beat.
    for (i, s) in [5, 7, 8].into_iter().enumerate() {
        let bell = Tone::new(penta(s), env(0.0015, 0.0, 0.5, 1.4), &BELL).amp(0.55);
        b.tone(0.25 + 0.32 * i as f64, -0.35 + 0.35 * i as f64, &bell);
    }
    // The coins: a cascade quickening, then thinning out.
    let mut seed = 0x00C0_1A5E_u64;
    let mut at = 0.25;
    for c in 0..18 {
        let s = 7 + (scatter(&mut seed) * 8.0) as i32;
        let pan = 1.4 * scatter(&mut seed) - 0.7;
        let amp = 0.5 + 0.5 * scatter(&mut seed);
        coin(&mut b, at, penta(s), pan, amp);
        let pace = if c < 10 {
            0.11 - 0.006 * f64::from(c)
        } else {
            0.05 + 0.012 * f64::from(c - 10)
        };
        at += pace;
    }
    // The win: a big major chord, spread wide.
    for (i, s) in [0, 3, 5, 7, 8].into_iter().enumerate() {
        let pl = Tone::new(penta(s), env(0.002, 0.05, 0.45, 1.1), &pluck(0.7)).amp(0.45);
        b.tone(1.6, -0.6 + 0.3 * i as f64, &pl);
        let bell = Tone::new(2.0 * penta(s), env(0.0015, 0.0, 0.35, 0.9), &BELL).amp(0.18);
        b.tone(1.6, 0.6 - 0.3 * i as f64, &bell);
    }
    b.room(0.35, 1.2);
    b
}

/// A gem popping in: `body` played into its 0.7 seconds, in a small room.
pub(super) fn gem(rate: u32, body: impl FnOnce(&mut Buf)) -> Buf {
    let mut b = Buf::new(rate, 0.7);
    body(&mut b);
    b.room(0.25, 0.5);
    b
}

/// A glassy plink on step `s` of the scale, at `at`, `pan` across, `amp`
/// loud.
pub(super) fn plink(b: &mut Buf, at: f64, s: i32, pan: f64, amp: f64) {
    let t = Tone::new(penta(s), env(0.001, 0.0, 0.12, 0.4), &GLASS).amp(amp);
    b.tone(at, pan, &t);
}

/// Good's gem: a chime.
pub fn gem_good(rate: u32) -> Buf {
    gem(rate, |b| {
        let chime = Tone::new(penta(6), env(0.0015, 0.0, 0.16, 0.45), &BELL);
        b.tone(0.0, 0.0, &chime);
    })
}

/// Great's: two bells.
pub fn gem_great(rate: u32) -> Buf {
    gem(rate, |b| {
        for (i, s) in [7, 9].into_iter().enumerate() {
            let bell = Tone::new(penta(s), env(0.0015, 0.0, 0.15, 0.4), &BELL);
            b.tone(0.035 * i as f64, -0.2 + 0.4 * i as f64, &bell);
        }
    })
}

/// Amazing's: a bright pair of plinks over a bell.
pub fn gem_amazing(rate: u32) -> Buf {
    gem(rate, |b| {
        plink(b, 0.0, 9, -0.2, 1.0);
        plink(b, 0.03, 11, 0.2, 0.8);
        let bell = Tone::new(penta(4), env(0.001, 0.0, 0.2, 0.5), &BELL).amp(0.4);
        b.tone(0.0, 0.0, &bell);
    })
}

/// Diva's: a sparkling run of plinks over a bell.
pub fn gem_diva(rate: u32) -> Buf {
    gem(rate, |b| {
        let mut seed = 0xD1A_u64;
        for g in 0..5 {
            let pan = 1.2 * scatter(&mut seed) - 0.6;
            plink(
                b,
                0.028 * f64::from(g),
                10 + g,
                pan,
                1.0 - 0.12 * f64::from(g),
            );
        }
        let bell = Tone::new(penta(5), env(0.001, 0.0, 0.3, 0.6), &BELL).amp(0.5);
        b.tone(0.0, 0.0, &bell);
    })
}
