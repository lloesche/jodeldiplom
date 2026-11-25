//! The original singer's phrases, the yardstick of the shine: the song's
//! vocals heard as the game hears a singer (the live pitch tracker, its
//! frame clock, the singer's key followed) and each phrase measured
//! ([`super::measure::yardstick`]).
//!
//! The yardstick is the least of three hearings, measure by measure (the
//! vocals at their own rate, and at a microphone's rate with the delay
//! [`REFERENCE_DELAY`] off either way): what the original surely reaches.
//! Each phrase's [`Par`] is what its everyday hearings reach against that
//! ([`par_hearings`]: other rates, the delay off, octave errors and the
//! pitch lost now and then): their least, less [`SPREAD`] of their spread; so that
//! the tiers' bars ([`super::Tier::of`]) are a share of what the original
//! itself reaches on it. A phrase whose par is under [`PAR_LEAST`] has no
//! shine: the bars there would let half a phrase sung be Diva.

use super::measure::{self, Measured, Yardstick};
use super::{shine, Frame, FRAME_LATE};
use crate::core::singing::{Notes, SungKey};
use jodel_rhythm::yin::{Yin, WINDOW};

/// The version of what a reference holds and how it is measured: raised
/// with any change to the measures, the hearings or the par, so a kept
/// one is measured again ([`crate::cache::SHINE`] is the file's layout).
/// 4: the live tracker hears the voice's band only
/// ([`jodel_rhythm::yin::BAND_LOW`]).
pub const VERSION: u32 = 4;

/// The delay the reference's microphone-rate hearings are off by (s).
pub const REFERENCE_DELAY: f64 = 0.03;
/// A microphone's rate.
const MIC_RATE: u32 = 48_000;

/// The least par shine a phrase has a shine with: under it the original
/// reaches so little of itself there that Diva's bar (90 % of the par)
/// would be near what half the phrase sung well reaches (its coverage
/// (0.5 + 0.1) / 1.1, times a quality of at most 1.07: 0.58; on the kept
/// songs it reaches 0.92 of the bar at most at this floor).
pub const PAR_LEAST: f64 = 0.70;

/// What the original reaches on a phrase in its everyday hearings: its
/// shine, and its pitch's share.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Par {
    /// Its shine.
    pub shine: f64,
    /// Its pitch's share.
    pub pitch: f64,
}

impl Default for Par {
    fn default() -> Par {
        Par {
            shine: 1.0,
            pitch: 1.0,
        }
    }
}

impl Par {
    /// The bars' scale: the original's shine where it is under 1.
    pub fn scale(&self) -> f64 {
        self.shine.clamp(0.0, 1.0)
    }

    /// The pitch bar's scale.
    pub fn pitch_scale(&self) -> f64 {
        self.pitch.clamp(0.0, 1.0)
    }
}

/// The original's phrase: its yardstick, and its par.
#[derive(Clone, Debug, PartialEq)]
pub struct Phrase {
    /// Its yardstick (the least of its hearings).
    pub yardstick: Yardstick,
    /// What its everyday hearings reach.
    pub par: Par,
}

/// The original's phrases, by sung line (none: no notes, or not heard).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reference {
    /// Each sung line's.
    pub phrases: Vec<Option<Phrase>>,
}

impl Reference {
    /// Line `l`'s phrase, where it has a shine to hold a singer to: heard
    /// enough ([`super::shines`]), its par at least [`PAR_LEAST`].
    pub fn phrase(&self, l: usize) -> Option<&Phrase> {
        self.phrases
            .get(l)?
            .as_ref()
            .filter(|p| super::shines(&p.yardstick) && p.par.shine >= PAR_LEAST)
    }

    /// The phrases with a shine.
    pub fn shining(&self) -> usize {
        (0..self.phrases.len())
            .filter(|&l| self.phrase(l).is_some())
            .count()
    }
}

/// A way of hearing the original: at `rate`, `late` seconds off, a share
/// `octave` of its frames with a pitch an octave off (up or down),
/// `dropped` of them lost, scattered by `seed`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hearing {
    /// The rate.
    pub rate: u32,
    /// How late.
    pub late: f64,
    /// The frames an octave off.
    pub octave: f64,
    /// The frames lost.
    pub dropped: f64,
    /// The scatter.
    pub seed: u64,
}

impl Hearing {
    /// At `rate`, `late` seconds off, nothing lost.
    pub const fn clean(rate: u32, late: f64) -> Hearing {
        Hearing {
            rate,
            late,
            octave: 0.0,
            dropped: 0.0,
            seed: 0,
        }
    }

    /// At a microphone's rate, `late` off, everyday errors: 3 % of the
    /// frames an octave off, 10 % lost.
    pub const fn everyday(late: f64, seed: u64) -> Hearing {
        Hearing {
            rate: MIC_RATE,
            late,
            octave: 0.03,
            dropped: 0.10,
            seed,
        }
    }
}

/// The hearings each phrase's par is the least of: [`PAR_RATES`] (those
/// with errors scattered [`PAR_SEEDS`] ways), and the everyday ones at a
/// microphone's rate with the delay off every [`PAR_STEP`] up to
/// [`PAR_MOST`] either way, [`PAR_SEEDS`] each (a phrase of few frames
/// swings from one hearing to the next: the least of many is what the
/// original surely reaches; once its pitch frames are in, a hearing costs
/// a few milliseconds).
pub fn par_hearings() -> Vec<Hearing> {
    let steps = (PAR_MOST / PAR_STEP).round() as i32;
    let mut out = Vec::new();
    for h in PAR_RATES {
        let scattered = h.octave > 0.0 || h.dropped > 0.0;
        let seeds = if scattered { PAR_SEEDS } else { 1 };
        out.extend((0..seeds).map(|s| Hearing {
            seed: h.seed + 500 * s,
            ..h
        }));
    }
    for k in -steps..=steps {
        for s in 0..PAR_SEEDS {
            let seed = 1000 + 97 * (k + steps) as u64 + 13 * s;
            out.push(Hearing::everyday(f64::from(k) * PAR_STEP, seed));
        }
    }
    out
}

/// The delays the par's everyday hearings are off by: every 5 ms, up to
/// 50 ms either way, two hearings each.
const PAR_STEP: f64 = 0.005;
const PAR_MOST: f64 = 0.05;
const PAR_SEEDS: u64 = 2;

/// The par's hearings at other rates, other errors.
pub const PAR_RATES: [Hearing; 7] = [
    Hearing::clean(16_000, 0.0),
    Hearing::everyday(-0.025, 13),
    Hearing::everyday(0.025, 17),
    Hearing {
        rate: 24_000,
        late: -0.045,
        octave: 0.04,
        dropped: 0.12,
        seed: 29,
    },
    Hearing {
        rate: 36_000,
        late: 0.045,
        octave: 0.04,
        dropped: 0.12,
        seed: 31,
    },
    Hearing {
        rate: MIC_RATE,
        late: 0.0,
        octave: 0.06,
        dropped: 0.2,
        seed: 37,
    },
    Hearing {
        rate: 32_000,
        late: 0.02,
        octave: 0.03,
        dropped: 0.1,
        seed: 41,
    },
];

/// A pitch frame as heard: when (seconds into the song), its pitch.
type Raw = (f64, Option<f32>);

/// The original's voice and the notes it is held to; its pitch frames at
/// each rate heard, worked out once.
pub struct Original<'a> {
    notes: &'a Notes,
    voice: &'a [f32],
    rate: u32,
    heard: Vec<(u32, Vec<Raw>)>,
}

impl<'a> Original<'a> {
    /// `voice` (mono, at `rate`) sung to `notes`.
    pub fn new(notes: &'a Notes, voice: &'a [f32], rate: u32) -> Original<'a> {
        Original {
            notes,
            voice,
            rate,
            heard: Vec::new(),
        }
    }

    /// The frames heard as `how` says (the voice's pitch frames at its
    /// rate, worked out once a rate), the singer's key followed.
    pub fn frames(&mut self, how: &Hearing) -> Vec<Frame> {
        let at = match self.heard.iter().position(|h| h.0 == how.rate) {
            Some(at) => at,
            None => {
                let frames = if how.rate == self.rate {
                    pitch_frames(self.voice, how.rate)
                } else {
                    pitch_frames(&resampled(self.voice, self.rate, how.rate), how.rate)
                };
                self.heard.push((how.rate, frames));
                self.heard.len() - 1
            }
        };
        keyed(self.notes, &corrupt(&self.heard[at].1, how))
    }

    /// The original heard as `how` says, each phrase's yardstick.
    pub fn yardsticks(&mut self, how: &Hearing) -> Vec<Option<Yardstick>> {
        let frames = self.frames(how);
        (0..self.notes.lines.len())
            .map(|l| measure::yardstick(self.notes, l, &frames))
            .collect()
    }

    /// The original heard as `how` says, each phrase measured against its
    /// yardstick in `against`.
    pub fn measure(
        &mut self,
        how: &Hearing,
        against: &[Option<Yardstick>],
    ) -> Vec<Option<Measured>> {
        let frames = self.frames(how);
        measure_all(self.notes, &frames, against)
    }

    /// The reference: the least of three hearings, and each phrase's par.
    pub fn reference(&mut self) -> Reference {
        let mut least_of = self.yardsticks(&Hearing::clean(self.rate, 0.0));
        for late in [-REFERENCE_DELAY, REFERENCE_DELAY] {
            let again = self.yardsticks(&Hearing::clean(MIC_RATE, late));
            least(&mut least_of, &again);
        }
        // Each phrase's shine and pitch share in each hearing: their least,
        // and their most.
        let mut seen: Vec<Option<[(f64, f64); 2]>> = vec![None; least_of.len()];
        for how in &par_hearings() {
            let heard = self.measure(how, &least_of);
            for (l, (p, y)) in heard.iter().zip(&least_of).enumerate() {
                let (Some(p), Some(y)) = (p, y) else { continue };
                let s = shine(p, y);
                let got = [s.value, s.pitch.unwrap_or(1.0)];
                let was = seen[l].get_or_insert([(f64::INFINITY, f64::NEG_INFINITY); 2]);
                for (w, g) in was.iter_mut().zip(got) {
                    *w = (w.0.min(g), w.1.max(g));
                }
            }
        }
        Reference {
            phrases: least_of
                .into_iter()
                .zip(seen)
                .map(|(y, seen)| {
                    y.map(|yardstick| Phrase {
                        yardstick,
                        par: seen.map_or_else(Par::default, |[s, p]| Par {
                            shine: surely(s),
                            pitch: surely(p),
                        }),
                    })
                })
                .collect(),
        }
    }
}

/// What the original surely reaches of a measure its everyday hearings
/// reached `least` to `most` of: the least, less [`SPREAD`] of that (a
/// phrase that swings much from one hearing to the next swings further in
/// others).
fn surely((least, most): (f64, f64)) -> f64 {
    least - SPREAD * (most - least)
}

/// The share of the spread the par is taken below the least.
const SPREAD: f64 = 0.5;

/// `voice` (mono, at `rate`) through the game's live pitch tracker: each
/// frame at the time in the song it was sung (its window's middle, the
/// frame clock's [`FRAME_LATE`] taken off).
pub fn pitch_frames(voice: &[f32], rate: u32) -> Vec<Raw> {
    let mut out = Vec::with_capacity(voice.len() / (rate as usize / 100).max(1) + 1);
    let mut yin = Yin::new(rate);
    for (i, &x) in voice.iter().enumerate() {
        yin.push(&[x], |cents| {
            let t = i as f64 / f64::from(rate) - WINDOW / 2.0 - FRAME_LATE;
            out.push((t, cents));
        });
    }
    out
}

/// `frames` with the singer's key followed as the game follows it (on
/// `notes`, frame by frame).
pub fn keyed(notes: &Notes, frames: &[Raw]) -> Vec<Frame> {
    let mut key = SungKey::default();
    frames
        .iter()
        .map(|&(t, cents)| {
            if let (Some(c), Some((_, note))) = (cents, notes.word_at(t)) {
                key.hear(f64::from(c), note);
            }
            Frame {
                t,
                cents,
                key: key.median(),
            }
        })
        .collect()
}

/// Every sung line of `notes` measured on `frames` against its yardstick
/// in `against` (by line).
pub fn measure_all(
    notes: &Notes,
    frames: &[Frame],
    against: &[Option<Yardstick>],
) -> Vec<Option<Measured>> {
    (0..notes.lines.len())
        .map(|l| {
            let y = against.get(l)?.as_ref()?;
            measure::phrase(notes, l, frames, y)
        })
        .collect()
}

/// `y` lowered, measure by measure, to what `again` (the original heard
/// another way) got: the words' frames, the long notes held, the pitch's
/// and the spread's errors, the range; its voicing (the timing's
/// yardstick) kept.
pub fn least(y: &mut [Option<Yardstick>], again: &[Option<Yardstick>]) {
    let high = |a: Option<f64>, b: Option<f64>| match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, _) => a,
    };
    for (y, a) in y.iter_mut().zip(again) {
        let (Some(y), Some(a)) = (y.as_mut(), a.as_ref()) else {
            continue;
        };
        for (h, g) in y.word_hits.iter_mut().zip(&a.word_hits) {
            *h = h.min(*g);
        }
        for (h, g) in y.held.iter_mut().zip(&a.held) {
            *h = h.min(*g);
        }
        y.voiced = y.voiced.min(a.voiced);
        y.off = high(y.off, a.off);
        y.spread = high(y.spread, a.spread);
        y.range = match (y.range, a.range) {
            (Some(x), Some(z)) => Some(x.min(z)),
            (x, _) => x,
        };
    }
}

/// `frames` heard as `how` says: `how.late` seconds late, octave errors and
/// frames lost scattered by its seed (the same each time).
fn corrupt(frames: &[(f64, Option<f32>)], how: &Hearing) -> Vec<(f64, Option<f32>)> {
    let mut rng = Lcg(how.seed);
    frames
        .iter()
        .map(|&(t, c)| {
            let t = t + how.late;
            let Some(c) = c else {
                return (t, None);
            };
            if how.octave <= 0.0 && how.dropped <= 0.0 {
                return (t, Some(c));
            }
            let x = rng.next();
            if x < how.dropped {
                (t, None)
            } else if x < how.dropped + how.octave {
                let up = rng.next() < 0.5;
                (t, Some(if up { c + 1200.0 } else { c - 1200.0 }))
            } else {
                (t, Some(c))
            }
        })
        .collect()
}

/// `voice` resampled from `from` to `to` Hz.
fn resampled(voice: &[f32], from: u32, to: u32) -> Vec<f32> {
    let mut r = jodel_playback::Resampler::new(from, to, 1);
    let input: Vec<f64> = voice.iter().map(|&x| f64::from(x)).collect();
    let mut out = Vec::new();
    r.process_f64(&input, &mut out);
    r.finish_f64(&mut out);
    out.into_iter().map(|v| v as f32).collect()
}

/// A small random number generator, the same each run.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}
