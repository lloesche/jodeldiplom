//! The result screen's own sounds, made ahead here ([`super::synth`]) at
//! the desk's rate and kept in its bank: what the game's synth does not
//! play live ([`super::conductor`]), the diploma's paper and pen, its
//! stamps and its tunes:
//!
//! - [`Sound::Medal`]: the medal's seal pressed down and its stinger, and
//!   [`Sound::NewBest`]: a new best's stamp and fanfare
//!   ([`super::cheers`]).
//! - [`Sound::Diploma`]: the diploma's own, from its paper unrolling to
//!   the examiner's signature ([`super::diploma`]).
//!
//! Each is set to its own peak ([`Sound::peak_db`]), well under the
//! song's; the Effects strip on the desk turns them all up or down. The
//! bank holds them in one list, kind by kind ([`Sound::all`]); a sound's
//! place is where it is in it.

use std::collections::HashMap;
use std::sync::LazyLock;

use super::cheers::{self, MEDALS};
use super::diploma::Diploma;
use super::synth::{Buf, Envelope, Partial, Tone};
use crate::core::singing::Medal;

/// A sound of the result screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sound {
    /// The medal stamped down, and its stinger.
    Medal(Medal),
    /// A new best.
    NewBest,
    /// One of the diploma's own.
    Diploma(Diploma),
}

/// The bank: every sound, in its order.
static BANK: LazyLock<Vec<Sound>> = LazyLock::new(|| {
    let mut all: Vec<Sound> = MEDALS.map(Sound::Medal).to_vec();
    all.push(Sound::NewBest);
    all.extend(Diploma::ALL.map(Sound::Diploma));
    all
});

/// Each sound's place in the bank.
static PLACES: LazyLock<HashMap<Sound, usize>> =
    LazyLock::new(|| BANK.iter().enumerate().map(|(k, &s)| (s, k)).collect());

impl Sound {
    /// How many there are.
    pub fn count() -> usize {
        BANK.len()
    }

    /// Every one, in the order of [`Sound::index`].
    pub fn all() -> Vec<Sound> {
        BANK.clone()
    }

    /// Where it is in the bank ([`Sound::all`]'s order).
    pub fn index(self) -> usize {
        PLACES[&self]
    }

    /// Its name, for its file when the sounds are written out to listen
    /// to.
    pub fn name(self) -> String {
        match self {
            Sound::Medal(m) => format!("medal-{}", m.name().to_lowercase()),
            Sound::NewBest => "new-best".into(),
            Sound::Diploma(d) => d.name(),
        }
    }

    /// The peak it is made at, dBFS: under the song, the bigger moments a
    /// little louder.
    pub fn peak_db(self) -> f64 {
        match self {
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
            Sound::Medal(m) => cheers::medal(rate, m),
            Sound::NewBest => cheers::new_best(rate),
            Sound::Diploma(d) => d.render(rate),
        };
        b.fade_out(0.01);
        b.peak_at(self.peak_db());
        b
    }
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

/// A brass-ish note: a saw's partials, its brightness swelling in.
pub(super) fn brass(freq: f64, hold: f64, length: f64) -> Tone {
    let partials: Vec<Partial> = (1..=10)
        .map(|n| Partial::new(f64::from(n), 1.0 / f64::from(n), f64::INFINITY))
        .collect();
    Tone::new(freq, env(0.015, hold, 0.25, length), &partials)
        .bright(700.0, 3_500.0, 0.04)
        .vibrato(5.5, 8.0, 0.25)
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
        assert_eq!(all.len(), Sound::count());
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
    fn the_scale_goes_up() {
        assert!((penta(0) - 523.251).abs() < 1e-3);
        assert!((penta(5) - 2.0 * 523.251).abs() < 1e-3);
        assert!((penta(-5) - 523.251 / 2.0).abs() < 1e-3);
        for s in -6..12 {
            assert!(penta(s + 1) > penta(s));
        }
    }
}
