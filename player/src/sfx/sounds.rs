//! The game's sounds, each made here ([`super::synth`]) at the desk's rate:
//!
//! - [`Sound::Pop`]: a bubble bursting. A wet "bloop": a sine whose pitch
//!   jumps up as it sounds (as a bubble's ring does as it closes), a
//!   droplet after it two steps up, a tick of bright noise for the snap
//!   and a soft thump under it. One for each step of a major pentatonic scale
//!   ([`STEPS`]), so pops of notes going up and down play a little tune.
//! - [`Sound::Sparkle`]: the splash's glitter, on the same step two
//!   octaves up: four glassy grains scattered over 90 ms, across the
//!   sides, in a small room.
//! - [`Sound::StepUp`]: the streak going up a level: a quick rising major
//!   arpeggio, plucked, higher, longer and brighter at each level, a swish
//!   of noise rising under it.
//! - [`Sound::Break`]: the streak lost: a soft "aww", a mellow voice-like
//!   tone sliding down a fourth, darkening as it goes; quiet.
//! - [`Sound::Zap`]: a shot fired to rescue a word missed: a "pew", a
//!   bright buzzy tone diving three octaves in a tenth of a second, a
//!   sparkle of noise on its start, an echo of it an octave down. The
//!   word rescued pops as any ([`Sound::Pop`]) as the shot lands.
//! - [`Sound::Tier`]: a phrase's rating: Good a two-note chime, Great a
//!   bright arpeggio with glitter, Amazing a "ta-daa" in brass with a
//!   shimmer, Diva a jackpot (a cascade of coins over bells, a whoosh, and
//!   a big chord).
//! - The result screen's: the count-up's ticks, the gems' plinks, the
//!   medal's seal pressed down and its stinger, a new best's stamp and
//!   fanfare ([`super::cheers`]); and the diploma's own, from its paper
//!   unrolling to the examiner's signature ([`Sound::Diploma`],
//!   [`super::diploma`]).
//!
//! Each is set to its own peak ([`Sound::peak_db`]), well under the
//! song's; the Effects strip on the desk turns them all up or down.

use super::cheers::{self, MEDALS, TICK_STEPS};
use super::diploma::Diploma;
use super::synth::{scatter, Buf, Envelope, Noise, Partial, Tone};
use crate::core::shine::Tier;
use crate::core::singing::Medal;

/// The steps of the scale the pops and sparkles are on.
pub const STEPS: usize = 9;

/// The streak's levels a step up has a sound for (1: ×2, up to ×4).
pub const LEVELS: u32 = 3;

/// A sound of the game.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sound {
    /// A bubble bursting, on step k of the scale (0, lowest, to
    /// [`STEPS`] - 1).
    Pop(usize),
    /// Its splash's glitter, on step k.
    Sparkle(usize),
    /// The streak up to level k (1 to [`LEVELS`]).
    StepUp(u32),
    /// The streak lost.
    Break,
    /// A shot fired to rescue a word missed.
    Zap,
    /// A phrase rated.
    Tier(Tier),
    /// The result's points counting up a tick, on step k (0 to
    /// [`TICK_STEPS`] - 1).
    Tick(usize),
    /// A phrase's gem popping in, by its tier (none: a plain one).
    Gem(Option<Tier>),
    /// The medal stamped down, and its stinger.
    Medal(Medal),
    /// A new best.
    NewBest,
    /// One of the diploma's own.
    Diploma(Diploma),
}

impl Sound {
    /// How many there are.
    pub const COUNT: usize = Self::DIPLOMA + Diploma::ALL.len();
    /// Where the result screen's start in the bank.
    const RESULT: usize = 2 * STEPS + LEVELS as usize + 2 + Tier::ALL.len();
    /// Where the diploma's start.
    const DIPLOMA: usize = Self::RESULT + TICK_STEPS + 1 + Tier::ALL.len() + MEDALS.len() + 1;

    /// Every one, in the order of [`Sound::index`].
    pub fn all() -> Vec<Sound> {
        let mut all: Vec<Sound> = (0..STEPS).map(Sound::Pop).collect();
        all.extend((0..STEPS).map(Sound::Sparkle));
        all.extend((1..=LEVELS).map(Sound::StepUp));
        all.push(Sound::Break);
        all.push(Sound::Zap);
        all.extend(Tier::ALL.map(Sound::Tier));
        all.extend((0..TICK_STEPS).map(Sound::Tick));
        all.push(Sound::Gem(None));
        all.extend(Tier::ALL.map(|t| Sound::Gem(Some(t))));
        all.extend(MEDALS.map(Sound::Medal));
        all.push(Sound::NewBest);
        all.extend(Diploma::ALL.map(Sound::Diploma));
        all
    }

    /// Where it is in the bank ([`Sound::all`]'s order). A step or a level
    /// out of range is the nearest there is.
    pub fn index(self) -> usize {
        let levels = LEVELS as usize;
        match self {
            Sound::Pop(k) => k.min(STEPS - 1),
            Sound::Sparkle(k) => STEPS + k.min(STEPS - 1),
            Sound::StepUp(l) => 2 * STEPS + (l.clamp(1, LEVELS) as usize - 1),
            Sound::Break => 2 * STEPS + levels,
            Sound::Zap => 2 * STEPS + levels + 1,
            Sound::Tier(t) => 2 * STEPS + levels + 2 + tier_index(t),
            Sound::Tick(k) => Self::RESULT + k.min(TICK_STEPS - 1),
            Sound::Gem(None) => Self::RESULT + TICK_STEPS,
            Sound::Gem(Some(t)) => Self::RESULT + TICK_STEPS + 1 + tier_index(t),
            Sound::Medal(m) => {
                let k = MEDALS.iter().position(|&x| x == m).unwrap_or(0);
                Self::RESULT + TICK_STEPS + 1 + Tier::ALL.len() + k
            }
            Sound::NewBest => Self::DIPLOMA - 1,
            Sound::Diploma(d) => Self::DIPLOMA + d.index(),
        }
    }

    /// Its name, for its file when the sounds are written out to listen
    /// to.
    pub fn name(self) -> String {
        match self {
            Sound::Pop(k) => format!("pop-{k}"),
            Sound::Sparkle(k) => format!("sparkle-{k}"),
            Sound::StepUp(l) => format!("step-up-{l}"),
            Sound::Break => "break".into(),
            Sound::Zap => "zap".into(),
            Sound::Tier(t) => format!("tier-{}", t.name().to_lowercase()),
            Sound::Tick(k) => format!("tick-{k}"),
            Sound::Gem(None) => "gem".into(),
            Sound::Gem(Some(t)) => format!("gem-{}", t.name().to_lowercase()),
            Sound::Medal(m) => format!("medal-{}", m.name().to_lowercase()),
            Sound::NewBest => "new-best".into(),
            Sound::Diploma(d) => d.name(),
        }
    }

    /// The peak it is made at, dBFS: under the song, the bigger moments a
    /// little louder.
    pub fn peak_db(self) -> f64 {
        match self {
            Sound::Pop(_) => -13.0,
            Sound::Sparkle(_) => -25.0,
            Sound::StepUp(l) => -16.5 + f64::from(l.clamp(1, LEVELS)),
            Sound::Break => -22.0,
            Sound::Zap => -17.0,
            Sound::Tier(Tier::Good) => -18.0,
            Sound::Tier(Tier::Great) => -15.5,
            Sound::Tier(Tier::Amazing) => -13.5,
            Sound::Tier(Tier::Diva) => -12.0,
            Sound::Tick(_) => -24.0,
            Sound::Gem(None) => -26.0,
            Sound::Gem(Some(t)) => -22.0 + 1.5 * tier_index(t) as f64,
            Sound::Medal(Medal::Bronze) => -14.0,
            Sound::Medal(Medal::Silver) => -13.0,
            Sound::Medal(Medal::Gold) => -12.0,
            Sound::NewBest => -12.5,
            Sound::Diploma(d) => d.peak_db(),
        }
    }

    /// It, made at `rate`: stereo.
    pub fn render(self, rate: u32) -> Buf {
        let mut b = match self {
            Sound::Pop(k) => pop(rate, k),
            Sound::Sparkle(k) => sparkle(rate, k),
            Sound::StepUp(l) => step_up(rate, l.clamp(1, LEVELS)),
            Sound::Break => aww(rate),
            Sound::Zap => zap(rate),
            Sound::Tier(Tier::Good) => good(rate),
            Sound::Tier(Tier::Great) => great(rate),
            Sound::Tier(Tier::Amazing) => amazing(rate),
            Sound::Tier(Tier::Diva) => diva(rate),
            Sound::Tick(k) => cheers::tick(rate, k.min(TICK_STEPS - 1)),
            Sound::Gem(t) => cheers::gem(rate, t),
            Sound::Medal(m) => cheers::medal(rate, m),
            Sound::NewBest => cheers::new_best(rate),
            Sound::Diploma(d) => d.render(rate),
        };
        b.fade_out(0.01);
        b.peak_at(self.peak_db());
        b
    }
}

/// Where `t` is among the tiers, Good first.
fn tier_index(t: Tier) -> usize {
    Tier::ALL.iter().position(|&x| x == t).unwrap_or(0)
}

/// Step `s` of the major pentatonic scale on C5 (0: C5; 5: C6; below 0
/// under it), in Hz.
pub fn penta(s: i32) -> f64 {
    const STEP: [i32; 5] = [0, 2, 4, 7, 9];
    let semis = 12 * s.div_euclid(5) + STEP[s.rem_euclid(5) as usize];
    523.251 * 2f64.powf(f64::from(semis) / 12.0)
}

pub(super) fn env(attack: f64, hold: f64, decay: f64, length: f64) -> Envelope {
    Envelope {
        attack,
        hold,
        decay,
        length,
    }
}

/// A sine, and a little of its second harmonic.
const ROUND: [Partial; 2] = [
    Partial::new(1.0, 1.0, f64::INFINITY),
    Partial::new(2.0, 0.12, 0.02),
];

/// A plucked tone's partials: the higher ones die first, `bright` (0 to 1)
/// how much of them there is.
pub(super) fn pluck(bright: f64) -> Vec<Partial> {
    (1..=6)
        .map(|n| {
            let n = f64::from(n);
            Partial::new(n, bright.powf(n - 1.0) / n, 0.25 / n)
        })
        .collect()
}

/// A bell's partials (inharmonic, as a struck bar's).
pub(super) const BELL: [Partial; 4] = [
    Partial::new(1.0, 1.0, f64::INFINITY),
    Partial::new(2.0, 0.35, 0.4),
    Partial::new(2.76, 0.25, 0.25),
    Partial::new(5.4, 0.12, 0.1),
];

/// A glassy grain: a sine and a high inharmonic overtone.
pub(super) const GLASS: [Partial; 2] = [
    Partial::new(1.0, 1.0, f64::INFINITY),
    Partial::new(4.07, 0.18, 0.03),
];

fn pop(rate: u32, k: usize) -> Buf {
    // It rings on its step of the scale, jumping up a fifth to it.
    let note = penta(k as i32);
    let f0 = note / 1.5;
    let mut b = Buf::new(rate, 0.16);
    // The ring: up to its note within its first 12 ms.
    let ring = Tone::new(f0, env(0.0008, 0.004, 0.03, 0.12), &ROUND).glide(1.5, 0.012);
    b.tone(0.0, 0.0, &ring);
    // A droplet after it, two steps up, rising onto its own.
    let drop = Tone::new(
        penta(k as i32 + 2) / 1.3,
        env(0.0008, 0.002, 0.018, 0.07),
        &ROUND,
    )
    .glide(1.3, 0.01)
    .amp(0.35);
    b.tone(0.022, 0.0, &drop);
    // The snap: a tick of bright noise.
    b.noise(
        0.0,
        0.0,
        &Noise {
            from: 4_500.0,
            to: 3_000.0,
            q: 1.2,
            envelope: env(0.0002, 0.0, 0.0015, 0.008),
            amp: 2.5,
            length: 0.008,
            seed: 0x9E37_79B9 + k as u64,
        },
    );
    // A soft thump under it: juice.
    let thump = Tone::new(0.5 * f0, env(0.001, 0.0, 0.012, 0.05), &ROUND[..1])
        .glide(0.8, 0.02)
        .amp(0.45);
    b.tone(0.0, 0.0, &thump);
    b
}

/// A buzzy tone's partials: odd ones, as a square wave's, the higher
/// dying first.
fn buzz() -> Vec<Partial> {
    [1.0, 3.0, 5.0, 7.0, 9.0]
        .map(|n: f64| Partial::new(n, 1.0 / n, 0.12 / n))
        .to_vec()
}

fn zap(rate: u32) -> Buf {
    let mut b = Buf::new(rate, 0.24);
    // The "pew": diving from high, its brightness closing as it goes.
    let pew = Tone::new(2_400.0, env(0.0015, 0.01, 0.05, 0.14), &buzz())
        .glide(0.12, 0.035)
        .bright(9_000.0, 1_800.0, 0.05);
    b.tone(0.0, 0.0, &pew);
    // An echo an octave down, a moment later, softer.
    let echo = Tone::new(1_200.0, env(0.002, 0.0, 0.05, 0.12), &buzz())
        .glide(0.15, 0.04)
        .bright(5_000.0, 1_200.0, 0.05)
        .amp(0.35);
    b.tone(0.03, 0.0, &echo);
    // A sparkle of noise on its start.
    b.noise(
        0.0,
        0.0,
        &Noise {
            from: 7_000.0,
            to: 3_500.0,
            q: 1.5,
            envelope: env(0.0005, 0.0, 0.01, 0.03),
            amp: 0.6,
            length: 0.03,
            seed: 0x2A9_u64,
        },
    );
    b.room(0.2, 0.2);
    b
}

fn sparkle(rate: u32, k: usize) -> Buf {
    let mut b = Buf::new(rate, 0.48);
    let base = k as i32 + 5;
    let mut seed = 0x5151_u64 + k as u64;
    for (g, (at, step)) in [(0.0, 5), (0.022, 7), (0.05, 8), (0.088, 10)]
        .into_iter()
        .enumerate()
    {
        let f = penta(base + step - 5);
        let pan = if g % 2 == 0 { -0.5 } else { 0.5 } * (0.6 + 0.4 * scatter(&mut seed));
        let grain = Tone::new(f, env(0.001, 0.0, 0.09, 0.3), &GLASS).amp(1.0 - 0.15 * g as f64);
        b.tone(at, pan, &grain);
    }
    b.room(0.35, 0.5);
    b
}

fn step_up(rate: u32, level: u32) -> Buf {
    let l = f64::from(level);
    let mut b = Buf::new(rate, 0.55 + 0.1 * l);
    // A major chord, a whole tone higher each level, one note more each
    // level above the first.
    let root = 523.251 * 2f64.powf(2.0 * (l - 1.0) / 12.0);
    let ratios = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0];
    let notes = 3 + level as usize;
    let bright = 0.35 + 0.15 * l;
    let gap = 0.042;
    for (i, r) in ratios.iter().take(notes).enumerate() {
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
    // A swish rising under it.
    b.noise(
        0.0,
        0.0,
        &Noise {
            from: 1_200.0,
            to: 5_000.0 + 1_500.0 * l,
            q: 2.0,
            envelope: env(0.12, 0.0, 0.05, 0.25),
            amp: 0.08 + 0.04 * l,
            length: 0.25,
            seed: 0xC0FFEE + u64::from(level),
        },
    );
    b.room(0.2, 0.6);
    b
}

fn aww(rate: u32) -> Buf {
    let mut b = Buf::new(rate, 0.75);
    // A soft voice: a saw's partials, rounded off, sliding down a fourth
    // from G4 and darkening as it goes (an "ah" closing to an "oo").
    let partials: Vec<Partial> = (1..=8)
        .map(|n| Partial::new(f64::from(n), 1.0 / f64::from(n), f64::INFINITY))
        .collect();
    let voice = Tone::new(392.0, env(0.04, 0.12, 0.25, 0.65), &partials)
        .glide(0.75, 0.18)
        .vibrato(5.0, 18.0, 0.3)
        .bright(1_500.0, 550.0, 0.25);
    b.tone(0.0, 0.0, &voice);
    // A third under it, quieter, for warmth.
    let under = Tone::new(311.1, env(0.06, 0.1, 0.25, 0.6), &partials)
        .glide(0.75, 0.2)
        .bright(1_000.0, 450.0, 0.25)
        .amp(0.4);
    b.tone(0.02, 0.0, &under);
    b.room(0.15, 0.4);
    b
}

fn good(rate: u32) -> Buf {
    let mut b = Buf::new(rate, 0.8);
    for (i, s) in [3, 5].into_iter().enumerate() {
        let chime = Tone::new(penta(s), env(0.0015, 0.0, 0.3, 0.6), &BELL);
        b.tone(0.09 * i as f64, -0.2 + 0.4 * i as f64, &chime);
    }
    b.room(0.25, 0.7);
    b
}

fn great(rate: u32) -> Buf {
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

/// A brass-ish note: a saw's partials, its brightness swelling in.
pub(super) fn brass(freq: f64, hold: f64, length: f64) -> Tone {
    let partials: Vec<Partial> = (1..=10)
        .map(|n| Partial::new(f64::from(n), 1.0 / f64::from(n), f64::INFINITY))
        .collect();
    Tone::new(freq, env(0.015, hold, 0.25, length), &partials)
        .bright(700.0, 3_500.0, 0.04)
        .vibrato(5.5, 8.0, 0.25)
}

fn amazing(rate: u32) -> Buf {
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

fn diva(rate: u32) -> Buf {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Each sound in its place in the bank, under a name of its own; each
    /// set to its peak, under the song's (none over -12 dBFS), ending on
    /// silence.
    #[test]
    fn every_sound_has_its_place_and_its_peak() {
        let all = Sound::all();
        assert_eq!(all.len(), Sound::COUNT);
        let mut names = std::collections::HashSet::new();
        for (k, s) in all.iter().enumerate() {
            assert_eq!(s.index(), k, "{s:?}");
            assert!(names.insert(s.name()), "{s:?}");
            assert!(s.peak_db() <= -12.0, "{s:?}");
        }
        for s in all {
            let b = s.render(48_000);
            let peak = b.l.iter().chain(&b.r).fold(0.0f64, |m, v| m.max(v.abs()));
            let db = 20.0 * peak.log10();
            assert!((db - s.peak_db()).abs() < 0.01, "{s:?}: {db}");
            // It ends on silence (no click when it stops).
            assert!(b.l.last().unwrap().abs() < 1e-6 && b.r.last().unwrap().abs() < 1e-6);
        }
    }

    #[test]
    fn pops_go_up_the_scale() {
        assert!((penta(0) - 523.251).abs() < 1e-3);
        assert!((penta(5) - 2.0 * 523.251).abs() < 1e-3);
        assert!((penta(-5) - 523.251 / 2.0).abs() < 1e-3);
        for s in -6..12 {
            assert!(penta(s + 1) > penta(s));
        }
    }
}
