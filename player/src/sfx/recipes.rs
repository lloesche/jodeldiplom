//! The tiers' own sounds, each a recipe a row of the tier table names
//! with the peak it is made at ([`crate::tiers::TIERS`], [`Recipe`]),
//! made at the desk's rate like the game's others ([`super::sounds`]):
//!
//! - A phrase's chime ([`super::Sound::Tier`]): a two-note chime
//!   ([`good`]), a bright arpeggio with glitter ([`great`]), a brass hit
//!   with a crash and plucks running up out of it ([`awesome`]), a
//!   "ta-daa" in brass with a shimmer ([`amazing`]), a jackpot (a cascade
//!   of coins over bells, a whoosh, and a big chord: [`diva`]).
//! - Its gem popping in on the diploma ([`super::Sound::Gem`]): a glassy
//!   plink, higher and richer up the tiers (a chime, two bells, two bells
//!   and a plink, a bright pair with a shimmer, a sparkling run).
//!
//! And the step up to each rung of the multiplier (a rung a tier), bigger
//! up the rungs ([`step_up`], [`super::Sound::StepUp`]).

use super::sounds::{brass, env, penta, pluck, BELL, GLASS};
use super::synth::{scatter, Buf, Noise, Partial, Tone};
use crate::tiers::Tier;

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

/// Awesome's: a brass hit (a short F major, punchy, a crash of bright
/// noise on it) and plucks running up out of it.
pub fn awesome(rate: u32) -> Buf {
    let mut b = Buf::new(rate, 1.3);
    for (i, f) in [349.2, 440.0, 523.3, 698.5].into_iter().enumerate() {
        b.tone(0.0, -0.45 + 0.3 * i as f64, &brass(f, 0.06, 0.4).amp(0.45));
    }
    b.noise(
        0.0,
        0.0,
        &Noise {
            from: 6_000.0,
            to: 3_000.0,
            q: 0.8,
            envelope: env(0.002, 0.0, 0.12, 0.35),
            amp: 0.25,
            length: 0.35,
            seed: 0xA3E5,
        },
    );
    for (i, s) in [5, 7, 8, 10].into_iter().enumerate() {
        let pl = Tone::new(penta(s), env(0.002, 0.0, 0.2, 0.45), &pluck(0.65)).amp(0.4);
        b.tone(0.18 + 0.05 * i as f64, -0.3 + 0.2 * i as f64, &pl);
    }
    b.room(0.3, 0.9);
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

/// The notes of the step up's arpeggio over its root: a major chord up
/// two octaves.
const ARPEGGIO: [f64; 7] = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0];

/// How far up the multiplier's rungs rung `rung` is (1: the first's, ×2):
/// 0 at the first, 1 at the top's ([`Tier::COUNT`]).
fn up_the_rungs(rung: usize) -> f64 {
    let top = Tier::COUNT.max(2) - 1;
    (rung.saturating_sub(1) as f64 / top as f64).min(1.0)
}

/// The step up to rung `rung` of the multiplier (1: ×2): a quick rising
/// major arpeggio, plucked, a swish rising under it, glitter scattered
/// over its end. Each grows from the first rung to the top's
/// ([`up_the_rungs`]): its root (up a minor sixth), its notes (four to
/// the whole arpeggio), its brightness, its glitter, its swish and its
/// length; so does its loudness ([`step_up_peak_db`]).
pub fn step_up(rate: u32, rung: usize) -> Buf {
    let x = up_the_rungs(rung);
    let root = 523.251 * 2f64.powf(8.0 * x / 12.0);
    let notes = 4 + (3.0 * x).round() as usize;
    let bright = 0.5 + 0.45 * x;
    let glitter = (8.0 * x).round() as usize;
    let (swish_to, swish_amp) = (6_500.0 + 6_000.0 * x, 0.12 + 0.16 * x);
    let mut b = Buf::new(rate, 0.75 + 0.3 * x + 0.03 * glitter as f64);
    let gap = 0.042;
    for (i, r) in ARPEGGIO.iter().take(notes).enumerate() {
        let last = i + 1 == notes;
        let decay = if last { 0.22 } else { 0.07 };
        let length = if last { 0.45 } else { 0.18 };
        let mut tone = Tone::new(root * r, env(0.002, 0.0, decay, length), &pluck(bright));
        if last {
            tone = tone.vibrato(6.0, 12.0, 0.15);
        }
        let pan = -0.3 + 0.6 * i as f64 / (notes - 1) as f64;
        b.tone(gap * i as f64, pan, &tone);
    }
    let mut seed = 0x5EED_u64 + rung as u64;
    for g in 0..glitter {
        let at = gap * (notes - 1) as f64 + 0.03 * g as f64;
        let step = 10 + (g % 6) as i32;
        let grain = Tone::new(penta(step), env(0.001, 0.0, 0.08, 0.25), &GLASS)
            .amp(0.3 * (1.0 - 0.04 * g as f64));
        b.tone(at, 1.6 * scatter(&mut seed) - 0.8, &grain);
    }
    b.noise(
        0.0,
        0.0,
        &Noise {
            from: 1_200.0,
            to: swish_to,
            q: 2.0,
            envelope: env(0.12, 0.0, 0.05, 0.25),
            amp: swish_amp,
            length: 0.25,
            seed: 0xC0FFEE + rung as u64,
        },
    );
    b.room(0.2, 0.6);
    b
}

/// The peak of the step up to rung `rung`, dBFS: from -15.5 at the first
/// to -12.5 at the top's ([`up_the_rungs`]), under the top's jackpot.
pub fn step_up_peak_db(rung: usize) -> f64 {
    -15.5 + 3.0 * up_the_rungs(rung)
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

/// Awesome's: two bells and a plink over them.
pub fn gem_awesome(rate: u32) -> Buf {
    gem(rate, |b| {
        for (i, s) in [7, 9].into_iter().enumerate() {
            let bell = Tone::new(penta(s), env(0.0015, 0.0, 0.15, 0.4), &BELL);
            b.tone(0.035 * i as f64, -0.2 + 0.4 * i as f64, &bell);
        }
        plink(b, 0.07, 11, 0.0, 0.7);
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
