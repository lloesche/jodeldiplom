//! The announcer: the slam's words called out, sung in the song's key on
//! its grid ("di-VA!", "times thir-ty-TWO!"), by one voice of our synth
//! (a formant voice: [`Atom::Singer`]) singing legato, its vowels the
//! notes' own shapes, its consonants short noises in a band
//! ([`Atom::Burst`], [`Atom::Hiss`]) or the voice humming with its mouth
//! closed.
//!
//! A word is its syllables ([`Syllable`]), each a vowel (or a diphthong,
//! gliding from one vowel to another) between the consonants before and
//! after it, so many eighths long, on a degree of the key's pentatonic over
//! the call's own tonic ([`HOME`]). The first vowel lands on the frame the
//! slam lands on (the stinger's hit for a rung, the jackpot's downbeat for a
//! phrase), each next one so many eighths on, a consonant before a vowel
//! leading into it; the last vowel holds, a vibrato coming in, and falls a
//! step down the key as it ends. A call over a sung line is sung softer
//! ([`DUCK`]); with no key known it is not sung at all (the voice is
//! pitched).

use super::palette::Atom;
use super::sounds::{Note, Score};
use crate::tiers::Tier;

/// The lowest the call's tonic is put at (C3): its melody, from the fifth
/// over it to the octave and a step, sits where an alto sings (G3 to C#5).
pub(super) const HOME: i32 = 48;

/// How much softer a call over a sung line is sung (about -8 dB).
pub(super) const DUCK: f32 = 0.4;

/// How long before its end the last vowel falls (a step down the key),
/// seconds.
const FALL_FOR: f64 = 0.09;

/// How loud the voice hums through a voiced consonant ("m", "v").
const HUM: f32 = 0.35;

/// How loud an open vowel ("a", "aw", "uh": shapes nearer "a" than
/// [`OPEN`]) is sung against a closed one: it rings louder through the
/// voice's formants (about 6 dB, measured), so the words come out alike.
const OPEN_LEVEL: f32 = 0.5;
const OPEN: f32 = 0.4;

/// How loud a vowel of `shape` is sung.
fn loudness(shape: f32) -> f32 {
    if shape.abs() < OPEN {
        OPEN_LEVEL
    } else {
        1.0
    }
}

/// A vowel as a note's shape on the voice: -1 "i", -0.5 "e", 0 "a", 0.5
/// "o", 1 "u" (the formant voice's vowels, "a" its own), and between.
const I: f32 = -1.0;
const IH: f32 = -0.84;
const EH: f32 = -0.5;
const UH: f32 = -0.2;
const A: f32 = 0.0;
const AW: f32 = 0.3;
const OO: f32 = 0.85;
const U: f32 = 1.0;

/// A consonant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Consonant {
    /// A stop ("t", "d", "k", "g"): a burst of noise in a band about
    /// `band` (a MIDI note), `lead` seconds before the vowel it opens (the
    /// longer, the breathier), or after the vowel it closes.
    Stop {
        /// The band's middle.
        band: f32,
        /// How far it comes before its vowel, seconds.
        lead: f64,
    },
    /// A fricative ("s", "f", "th", "v", "z"): a hiss in a band about
    /// `band` for `length` seconds, at `level` (0 to 1), the voice humming
    /// through it where it is `voiced`.
    Hiss {
        /// The band's middle.
        band: f32,
        /// How long it hisses, seconds.
        length: f64,
        /// How loud.
        level: f32,
        /// Whether the voice hums through it.
        voiced: bool,
    },
    /// A nasal ("m", "n", "ng"): the voice humming, its mouth closed, for
    /// `length` seconds.
    Hum {
        /// How long, seconds.
        length: f64,
    },
}

impl Consonant {
    /// How long it takes, seconds.
    fn length(self) -> f64 {
        match self {
            Consonant::Stop { lead, .. } => lead,
            Consonant::Hiss { length, .. } | Consonant::Hum { length } => length,
        }
    }
}

const T: Consonant = Consonant::Stop {
    band: 109.0,
    lead: 0.04,
};
const K: Consonant = Consonant::Stop {
    band: 96.0,
    lead: 0.035,
};
const D: Consonant = Consonant::Stop {
    band: 103.0,
    lead: 0.015,
};
const G: Consonant = Consonant::Stop {
    band: 92.0,
    lead: 0.015,
};
const S: Consonant = Consonant::Hiss {
    band: 115.5,
    length: 0.075,
    level: 1.0,
    voiced: false,
};
const Z: Consonant = Consonant::Hiss {
    band: 115.5,
    length: 0.06,
    level: 0.5,
    voiced: true,
};
const F: Consonant = Consonant::Hiss {
    band: 110.0,
    length: 0.06,
    level: 0.5,
    voiced: false,
};
const TH: Consonant = Consonant::Hiss {
    band: 106.0,
    length: 0.05,
    level: 0.4,
    voiced: false,
};
const V: Consonant = Consonant::Hiss {
    band: 106.0,
    length: 0.05,
    level: 0.3,
    voiced: true,
};
const M: Consonant = Consonant::Hum { length: 0.06 };
const N: Consonant = Consonant::Hum { length: 0.06 };
const NG: Consonant = Consonant::Hum { length: 0.07 };

/// A syllable: the consonants before its vowel, its vowel (and the one it
/// glides to, a diphthong's), those after it, how many eighths it lasts
/// (from its vowel to the next one's), the degree of the key it is sung on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Syllable {
    pub(super) before: &'static [Consonant],
    pub(super) vowel: f32,
    pub(super) to: Option<f32>,
    pub(super) after: &'static [Consonant],
    pub(super) eighths: u32,
    pub(super) degree: i32,
}

/// A word: its syllables.
pub(super) type Word = &'static [Syllable];

/// A syllable of `vowel` on `degree` for `eighths`, nothing round it.
const fn syl(vowel: f32, eighths: u32, degree: i32) -> Syllable {
    Syllable {
        before: &[],
        vowel,
        to: None,
        after: &[],
        eighths,
        degree,
    }
}

/// The same, gliding to `to`.
const fn glide(from: f32, to: f32, eighths: u32, degree: i32) -> Syllable {
    Syllable {
        to: Some(to),
        ..syl(from, eighths, degree)
    }
}

/// `s` with `before` and `after` round its vowel.
const fn with(before: &'static [Consonant], s: Syllable, after: &'static [Consonant]) -> Syllable {
    Syllable { before, after, ..s }
}

const GOOD: [Syllable; 1] = [with(&[G], syl(OO, 3, 5), &[D])];
const GREAT: [Syllable; 1] = [with(&[G], glide(EH, I, 3, 5), &[T])];
const AWESOME: [Syllable; 2] = [syl(AW, 2, 5), with(&[S], syl(UH, 2, 3), &[M])];
const AMAZING: [Syllable; 3] = [
    syl(UH, 1, 3),
    with(&[M], glide(EH, I, 2, 5), &[]),
    with(&[Z], syl(IH, 2, 4), &[NG]),
];
const DIVA: [Syllable; 2] = [with(&[D], syl(I, 1, 3), &[]), with(&[V], syl(A, 3, 5), &[])];

/// "times", before a multiplier's number.
const TIMES: Syllable = with(&[T], glide(A, I, 1, 3), &[M, Z]);
const TIMES_TWO: [Syllable; 2] = [TIMES, with(&[T], syl(U, 3, 5), &[])];
const TIMES_FOUR: [Syllable; 2] = [TIMES, with(&[F], syl(AW, 3, 5), &[])];
const TIMES_EIGHT: [Syllable; 2] = [TIMES, with(&[], glide(EH, I, 3, 5), &[T])];
const TIMES_SIXTEEN: [Syllable; 3] = [
    TIMES,
    with(&[S], syl(IH, 1, 4), &[K, S]),
    with(&[T], syl(I, 3, 5), &[N]),
];
const TIMES_THIRTY_TWO: [Syllable; 4] = [
    TIMES,
    with(&[TH], syl(UH, 1, 5), &[]),
    with(&[T], syl(I, 1, 4), &[]),
    with(&[T], syl(U, 3, 5), &[]),
];

/// The word a phrase reaching `tier` is called out with: its name (none
/// for a name it has no word for).
pub(super) fn name(tier: Tier) -> Option<Word> {
    Some(match tier.name() {
        "Good" => &GOOD,
        "Great" => &GREAT,
        "Awesome" => &AWESOME,
        "Amazing" => &AMAZING,
        "Diva" => &DIVA,
        _ => return None,
    })
}

/// The words a rung reached is called out with: "times" and its
/// multiplier (none for a number it has no word for).
pub(super) fn times(multiplier: u32) -> Option<Word> {
    Some(match multiplier {
        2 => &TIMES_TWO,
        4 => &TIMES_FOUR,
        8 => &TIMES_EIGHT,
        16 => &TIMES_SIXTEEN,
        32 => &TIMES_THIRTY_TWO,
        _ => return None,
    })
}

/// How long `word` sounds from its first vowel, frames, at `beat` frames a
/// beat and `rate` frames a second: its eighths, and the release after.
pub(super) fn span(word: Word, beat: f64, rate: f64) -> u64 {
    let eighths: u32 = word.iter().map(|s| s.eighths).sum();
    (f64::from(eighths) * beat / 2.0 + 0.15 * rate).round() as u64
}

/// `word` sung from `now` (nothing before it), its first vowel on `at`, at
/// `gain` (1: as loud as it is; [`DUCK`] over a sung line); with no key,
/// nothing.
pub(super) fn sing(s: &mut Score, now: u64, at: u64, word: Word, gain: f32) {
    let Some(key) = s.key else {
        return;
    };
    let home = key.home(HOME);
    let eighth = s.beat / 2.0;
    let mut vowel_at = at;
    for (k, syllable) in word.iter().enumerate() {
        let pitch = key.note(syllable.degree, home);
        // What leads into the vowel, in its order, ending on it.
        let lead: f64 = syllable.before.iter().map(|c| c.length()).sum();
        let mut t = vowel_at.saturating_sub(s.secs(lead)).max(now);
        for &c in syllable.before {
            let length = s.secs(c.length()).min(vowel_at.saturating_sub(t));
            consonant(s, t, length, c, pitch, gain, false);
            t += length;
        }
        let next = vowel_at + (f64::from(syllable.eighths) * eighth).round() as u64;
        let last = k + 1 == word.len();
        // Up to what closes it and leads into the next one.
        let closing: f64 = syllable.after.iter().map(|c| c.length()).sum();
        let opening: f64 = word
            .get(k + 1)
            .map_or(0.0, |n| n.before.iter().map(|c| c.length()).sum());
        let end = next
            .saturating_sub(s.secs(closing + opening))
            .max(vowel_at + s.secs(0.04));
        // The last one falls a step down the key as it ends, and what
        // closes it is hummed as low.
        let low = if last {
            key.note(syllable.degree - 1, home)
        } else {
            pitch
        };
        vowel(s, (vowel_at, end), syllable, (pitch, low), gain);
        let mut t = end;
        for &c in syllable.after {
            let length = s.secs(c.length());
            consonant(s, t, length, c, low, gain, true);
            t += length;
        }
        vowel_at = next;
    }
}

/// A syllable's vowel from `at` to `end` on `pitch` (a diphthong gliding
/// to its second vowel past halfway), falling to `low` as it ends where
/// that is lower (the last one's).
fn vowel(
    s: &mut Score,
    (at, end): (u64, u64),
    syllable: &Syllable,
    (pitch, low): (f32, f32),
    gain: f32,
) {
    let fall = if low < pitch {
        s.secs(FALL_FOR).min((end - at) / 2)
    } else {
        0
    };
    let held = end - at - fall;
    let sung = |at: u64, length: u64, shape: f32, pitch: f32| {
        Note::new(at, Atom::Singer, pitch)
            .vel(gain * loudness(shape))
            .shape(shape)
            .hold(length.max(1))
    };
    match syllable.to {
        Some(to) => {
            let half = held * 11 / 20;
            s.play(sung(at, half, syllable.vowel, pitch));
            s.play(sung(at + half, held - half, to, pitch));
        }
        None => s.play(sung(at, held, syllable.vowel, pitch)),
    }
    if fall > 0 {
        let shape = syllable.to.unwrap_or(syllable.vowel);
        let falling = sung(at + held, fall, shape, low);
        s.play(falling.vel(0.8 * gain * loudness(shape)));
    }
}

/// Consonant `c` from `at`, `length` frames, by a vowel on `pitch`; a stop
/// `closing` a vowel bursts as its closure ends.
fn consonant(
    s: &mut Score,
    at: u64,
    length: u64,
    c: Consonant,
    pitch: f32,
    gain: f32,
    closing: bool,
) {
    let hum = |s: &mut Score| {
        let note = Note::new(at, Atom::Singer, pitch).shape(U);
        s.play(note.vel(HUM * gain).hold(length.max(1)));
    };
    match c {
        Consonant::Stop { band, .. } => {
            let burst = if closing { at + length } else { at };
            s.play(Note::new(burst, Atom::Burst, band).vel(gain));
        }
        Consonant::Hiss {
            band,
            level,
            voiced,
            ..
        } => {
            let hiss = Note::new(at, Atom::Hiss, band).vel(level * gain);
            s.play(hiss.hold(length.max(1)));
            if voiced {
                hum(s);
            }
        }
        Consonant::Hum { .. } => hum(s),
    }
}
