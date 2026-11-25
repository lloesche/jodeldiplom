//! The diploma's own sounds ([`Diploma`]), made here at the desk's rate
//! like the game's others ([`super::sounds`]), each timed inside to what
//! the stage draws from its beat (the reveal's times and motions, shared
//! with the stage: [`crate::core::singing::Reveal`]'s, [`unrolled`] and
//! [`signed`]):
//!
//! - [`Diploma::Unroll`]: the sheet unrolling: a soft rustle of paper, as
//!   loud as the sheet's front is fast (quick, then slowing), crinkles in
//!   it, and a soft flump as it lies flat.
//! - [`Diploma::Star`]: a star by the heading popping on: a little glassy
//!   ping, the right one higher.
//! - [`Diploma::Title`]: the song's title landing: a bright "ta-da" in
//!   bells and plucks, glitter off its ends as they sparkle.
//! - [`Diploma::Written`]: an entry written in by hand (the words hit, the
//!   best streak): the pen scratching over its writing time, lifted between
//!   words, then a round "pop" as its value lands, higher for each entry
//!   down the sheet. [`Diploma::Scribbled`]: the pen alone, for an entry
//!   with nothing to cheer for.
//! - [`Diploma::Sticker`]: a star sticker pressed on where no seal goes: a
//!   sticky "plop" and a glint.
//! - The verdicts without a medal, the grade written in under each:
//!   [`Diploma::Cheer`] a friendly little jingle rising to a chord,
//!   [`Diploma::Whole`] two calm chimes over a soft chord, and
//!   [`Diploma::Snooze`], where nothing was scored, a music box playing
//!   "Silent night" softly, sleepily.
//! - [`Diploma::Stamp`]: a rubber stamp slapped on: a rush of air as it
//!   comes down, the thunk of it landing, the paper flapping under it.
//! - [`Diploma::Signature`]: the examiner signing: a pen scratching along
//!   the signature's very stroke ([`signature_point`], at its pace),
//!   louder and higher where it goes fast, a flick as the pen comes off.
//! - [`Diploma::Date`]: the date written in after it, a tap of the pen on
//!   its end.
//! - [`Diploma::Tap`]: the best line shown calmly: a soft tap of the pen.
//!
//! The pen on paper is [`super::pen`]'s. The count-up's ticks, the points
//! all counted, the gems and the best streak's rung are the game's synth's,
//! as its other sounds are ([`super::conductor::Diploma`]).

use super::pen::{burst, handwriting, scratch, tap, DATE_SAMPLE};
use super::sounds::{env, penta, pluck, BELL, GLASS};
use super::synth::{scatter, white, Biquad, Buf, Partial, Tone};
use crate::core::singing::signature::signature_point;
use crate::core::singing::{
    signed, unrolled, Medal, Side, Verdict, DATE_FROM, DATING, GRADE_WRITE, SIGNING, STAMP_DOWN,
    STREAK_WRITE, UNROLL, WORDS_FILL,
};

/// An entry written in by hand below the points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Entry {
    /// The words hit.
    Words,
    /// The best streak.
    Streak,
}

impl Entry {
    /// How long it takes to write (the stage's), a sample of what is
    /// written there (its words part where the pen lifts), and the note
    /// its value lands on (a step up the scale down the sheet).
    fn writing(self) -> (f64, &'static str, i32) {
        match self {
            Entry::Words => (WORDS_FILL, "40 of 45", 2),
            Entry::Streak => (STREAK_WRITE, "17 in a row", 4),
        }
    }
}

/// A sound of the diploma.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Diploma {
    /// The sheet unrolling.
    Unroll,
    /// A star by the heading popping on.
    Star(Side),
    /// The song's title landing.
    Title,
    /// An entry written in, its value landing.
    Written(Entry),
    /// An entry written in, nothing to cheer for.
    Scribbled(Entry),
    /// A star sticker pressed on where no seal goes.
    Sticker,
    /// No medal: a cheer.
    Cheer,
    /// Not sung from its start.
    Whole,
    /// Nothing scored.
    Snooze,
    /// A rubber stamp slapped on.
    Stamp,
    /// The examiner signing.
    Signature,
    /// The date written in.
    Date,
    /// A tap of the pen.
    Tap,
}

impl Diploma {
    /// Every one, in the bank's order ([`Diploma::index`]).
    pub const ALL: [Diploma; 16] = [
        Diploma::Unroll,
        Diploma::Star(Side::Left),
        Diploma::Star(Side::Right),
        Diploma::Title,
        Diploma::Written(Entry::Words),
        Diploma::Written(Entry::Streak),
        Diploma::Scribbled(Entry::Words),
        Diploma::Scribbled(Entry::Streak),
        Diploma::Sticker,
        Diploma::Cheer,
        Diploma::Whole,
        Diploma::Snooze,
        Diploma::Stamp,
        Diploma::Signature,
        Diploma::Date,
        Diploma::Tap,
    ];

    /// Where it is among them ([`Diploma::ALL`], which has every one).
    pub fn index(self) -> usize {
        Diploma::ALL
            .iter()
            .position(|&d| d == self)
            .expect("every one is in ALL")
    }

    /// Its name, for its file.
    pub fn name(self) -> String {
        let entry = |e: Entry| match e {
            Entry::Words => "words",
            Entry::Streak => "streak",
        };
        match self {
            Diploma::Unroll => "diploma-unroll".into(),
            Diploma::Star(Side::Left) => "diploma-star-left".into(),
            Diploma::Star(Side::Right) => "diploma-star-right".into(),
            Diploma::Title => "diploma-title".into(),
            Diploma::Written(e) => format!("diploma-written-{}", entry(e)),
            Diploma::Scribbled(e) => format!("diploma-scribbled-{}", entry(e)),
            Diploma::Sticker => "diploma-sticker".into(),
            Diploma::Cheer => "diploma-cheer".into(),
            Diploma::Whole => "diploma-whole".into(),
            Diploma::Snooze => "diploma-snooze".into(),
            Diploma::Stamp => "diploma-stamp".into(),
            Diploma::Signature => "diploma-signature".into(),
            Diploma::Date => "diploma-date".into(),
            Diploma::Tap => "diploma-tap".into(),
        }
    }

    /// The peak it is made at, dBFS: under the song, the paper's own
    /// sounds the softest, the bigger moments a little louder (a verdict
    /// without a medal under every medal's).
    pub fn peak_db(self) -> f64 {
        match self {
            Diploma::Unroll | Diploma::Star(_) => -25.0,
            Diploma::Title => -19.0,
            Diploma::Written(_) => -20.0,
            Diploma::Scribbled(_) => -29.0,
            Diploma::Sticker => -21.0,
            Diploma::Cheer => -17.0,
            Diploma::Whole => -19.0,
            Diploma::Snooze => -21.0,
            Diploma::Stamp => -16.0,
            Diploma::Signature => -24.0,
            Diploma::Date => -27.0,
            Diploma::Tap => -28.0,
        }
    }

    /// It, made at `rate` (not yet set to its peak).
    pub fn render(self, rate: u32) -> Buf {
        match self {
            Diploma::Unroll => unroll(rate),
            Diploma::Star(side) => star(rate, side),
            Diploma::Title => title(rate),
            Diploma::Written(e) => written(rate, e, true),
            Diploma::Scribbled(e) => written(rate, e, false),
            Diploma::Sticker => sticker(rate),
            Diploma::Cheer => cheer(rate),
            Diploma::Whole => whole(rate),
            Diploma::Snooze => snooze(rate),
            Diploma::Stamp => {
                let mut b = Buf::new(rate, 0.6);
                rubber_stamp(&mut b, 0.0);
                b.room(0.12, 0.25);
                b
            }
            Diploma::Signature => signature(rate),
            Diploma::Date => date(rate),
            Diploma::Tap => {
                let mut b = Buf::new(rate, 0.12);
                tap(&mut b, 0.0, 1.0);
                b
            }
        }
    }
}

/// How fast `share(t)` moves at `t` seconds (its share a second), from 1
/// ms either side.
fn pace(share: impl Fn(f64) -> f64, t: f64) -> f64 {
    let h = 0.001;
    (share(t + h) - share((t - h).max(0.0))) / (t + h - (t - h).max(0.0))
}

fn unroll(rate: u32) -> Buf {
    let mut b = Buf::new(rate, UNROLL + 0.3);
    // The sheet's front moves as the stage unrolls it ([`unrolled`]):
    // fast, then slowing to a stop; the rustle as loud as it goes.
    let fastest = pace(unrolled, 0.0);
    let front = |t: f64| pace(unrolled, t) / fastest;
    let mut seed = 0x9A9E_u64;
    let mut rustle = Biquad::band(900.0, 0.7, b.rate);
    let mut air = Biquad::band(320.0, 0.6, b.rate);
    let ease = 1.0 - (-1.0 / (0.004 * b.rate)).exp();
    let crinkles = 1.0 - (-2.0 * std::f64::consts::PI * 35.0 / b.rate).exp();
    let (mut crinkle, mut fade_in, mut k) = (0.0f64, 0.0f64, 0usize);
    let rate_f = b.rate;
    b.add(0.0, UNROLL, 0.0, |t| {
        fade_in += ease * (1.0 - fade_in);
        if k.is_multiple_of(32) {
            rustle = rustle.retuned(900.0 + 1_800.0 * (t / UNROLL), 0.7, rate_f);
        }
        k += 1;
        crinkle += crinkles * (white(&mut seed) - crinkle);
        let x = white(&mut seed);
        let v = front(t) * fade_in;
        v * ((0.6 + 5.0 * crinkle.abs()).min(1.5) * rustle.run(x) + 0.5 * air.run(x))
    });
    // Crinkles of the paper as it goes, more where it is quicker.
    let mut seed = 0xC41_u64;
    for _ in 0..16 {
        let at = UNROLL * 0.85 * scatter(&mut seed).powf(1.6);
        let f = 3_000.0 + 3_000.0 * scatter(&mut seed);
        let amp = 0.8 * front(at) + 0.15;
        let pan = 0.8 * scatter(&mut seed) - 0.4;
        b.noise(
            at,
            pan,
            &burst((f, f * 0.8), 2.5, (0.0003, 0.002, 0.008), amp),
        );
    }
    // It lies flat: a soft flump of air, the paper's edge settling.
    let sine = [Partial::new(1.0, 1.0, 1.0)];
    let flump = Tone::new(120.0, env(0.004, 0.0, 0.03, 0.12), &sine)
        .glide(0.7, 0.04)
        .amp(0.35);
    b.tone(UNROLL * 0.9, 0.0, &flump);
    let settle = burst((1_400.0, 900.0), 0.9, (0.003, 0.03, 0.09), 0.5);
    b.noise(UNROLL * 0.9, 0.0, &settle);
    b
}

fn star(rate: u32, side: Side) -> Buf {
    let mut b = Buf::new(rate, 0.5);
    let s = match side {
        Side::Left => 8,
        Side::Right => 10,
    };
    let ping = Tone::new(penta(s), env(0.0008, 0.0, 0.09, 0.35), &BELL);
    b.tone(0.0, 0.0, &ping);
    let glint = Tone::new(penta(s + 5), env(0.001, 0.0, 0.05, 0.2), &GLASS).amp(0.35);
    b.tone(0.03, 0.0, &glint);
    b.room(0.25, 0.4);
    b
}

fn title(rate: u32) -> Buf {
    let mut b = Buf::new(rate, 1.3);
    // "Ta": a short G; "da!": a C major chord, ringing.
    let ta = Tone::new(penta(3), env(0.0015, 0.02, 0.06, 0.14), &pluck(0.6)).amp(0.7);
    b.tone(0.0, 0.0, &ta);
    let ta_bell = Tone::new(penta(8), env(0.001, 0.0, 0.06, 0.14), &BELL).amp(0.25);
    b.tone(0.0, 0.0, &ta_bell);
    for (i, s) in [5, 7, 8].into_iter().enumerate() {
        let pan = -0.3 + 0.3 * i as f64;
        let da = Tone::new(penta(s), env(0.002, 0.04, 0.3, 0.8), &pluck(0.65)).amp(0.45);
        b.tone(0.11, pan, &da);
        let bell = Tone::new(2.0 * penta(s), env(0.0015, 0.0, 0.25, 0.6), &BELL).amp(0.2);
        b.tone(0.11, -pan, &bell);
    }
    // Glitter off its two ends as they sparkle (0.3 s after it lands).
    let mut seed = 0x717_u64;
    for g in 0..8 {
        let side = if g % 2 == 0 { -0.65 } else { 0.65 };
        let at = 0.3 + 0.03 * f64::from(g) + 0.01 * scatter(&mut seed);
        let grain = Tone::new(penta(11 + g % 4), env(0.001, 0.0, 0.07, 0.22), &GLASS).amp(0.3);
        b.tone(at, side, &grain);
    }
    b.room(0.3, 0.8);
    b
}

fn written(rate: u32, entry: Entry, lands: bool) -> Buf {
    let (length, sample, step) = entry.writing();
    let mut b = Buf::new(rate, length + if lands { 0.5 } else { 0.1 });
    let mut pen = Buf::new(rate, length + 0.1);
    handwriting(&mut pen, (0.0, length), sample, (0.0, 0x3417 + step as u64));
    pen.peak_at(0.0);
    b.mix(0.0, &pen, if lands { 0.3 } else { 1.0 });
    if lands {
        // The value in: a round pop rising onto its note, a soft thud of
        // the paper under it.
        let f = penta(step);
        let pop = Tone::new(f / 1.4, env(0.001, 0.008, 0.06, 0.25), &BELL[..2]).glide(1.4, 0.012);
        b.tone(length, 0.0, &pop);
        let sine = [Partial::new(1.0, 1.0, 1.0)];
        let thud = Tone::new(0.5 * f, env(0.001, 0.0, 0.025, 0.09), &sine)
            .glide(0.75, 0.02)
            .amp(0.5);
        b.tone(length, 0.0, &thud);
        let tock = burst((1_600.0, 1_100.0), 1.5, (0.0003, 0.004, 0.015), 0.6);
        b.noise(length, 0.0, &tock);
        b.room(0.18, 0.4);
    }
    b
}

fn sticker(rate: u32) -> Buf {
    let mut b = Buf::new(rate, 0.6);
    // Peeled off and pressed on: a sticky "tsk", a round plop dropping a
    // little as it settles, then a glint.
    b.noise(
        0.0,
        0.0,
        &burst((2_200.0, 1_500.0), 1.5, (0.0004, 0.004, 0.02), 0.6),
    );
    let plop = Tone::new(420.0, env(0.001, 0.01, 0.04, 0.14), &BELL[..2]).glide(0.7, 0.03);
    b.tone(0.004, 0.0, &plop);
    let glint = Tone::new(penta(10), env(0.001, 0.0, 0.08, 0.3), &GLASS).amp(0.3);
    b.tone(0.13, 0.2, &glint);
    b.room(0.2, 0.4);
    b
}

/// `verdict`'s grade ([`Verdict::grade`]) written in by hand as the
/// stage writes it ([`GRADE_WRITE`]), from the verdict's beat, at `amp`
/// of full scale.
pub fn grade(b: &mut Buf, verdict: Verdict, amp: f64) {
    let mut pen = Buf::new(b.rate as u32, GRADE_WRITE + 0.1);
    handwriting(&mut pen, (0.0, GRADE_WRITE), verdict.grade(), (0.0, 0x6EAD));
    pen.peak_at(0.0);
    b.mix(0.0, &pen, amp);
}

/// The grade of a medal written in under its seal and stinger.
pub fn medal_grade(b: &mut Buf, medal: Medal) {
    grade(b, Verdict::Medal(medal), 0.08);
}

fn cheer(rate: u32) -> Buf {
    let mut b = Buf::new(rate, 1.6);
    grade(&mut b, Verdict::Cheer, 0.12);
    // Up a little tune, G C E, to a warm C major chord with a bell on top.
    for (i, s) in [3, 5, 7].into_iter().enumerate() {
        let t = 0.06 + 0.09 * i as f64;
        let pl = Tone::new(penta(s), env(0.002, 0.0, 0.12, 0.3), &pluck(0.55)).amp(0.6);
        b.tone(t, -0.25 + 0.25 * i as f64, &pl);
    }
    for (i, s) in [5, 7, 8].into_iter().enumerate() {
        let pl = Tone::new(penta(s), env(0.003, 0.08, 0.35, 0.9), &pluck(0.5)).amp(0.4);
        b.tone(0.33, -0.3 + 0.3 * i as f64, &pl);
    }
    let bell = Tone::new(penta(10), env(0.0015, 0.0, 0.35, 0.9), &BELL).amp(0.25);
    b.tone(0.33, 0.0, &bell);
    b.room(0.3, 0.8);
    b
}

/// A soft chord of sines on `steps`, from `at`, `length` long.
fn pad(b: &mut Buf, at: f64, steps: &[i32], length: f64, amp: f64) {
    let sines = [
        Partial::new(1.0, 1.0, f64::INFINITY),
        Partial::new(2.0, 0.08, f64::INFINITY),
    ];
    for (i, &s) in steps.iter().enumerate() {
        let pan = if steps.len() > 1 {
            -0.3 + 0.6 * i as f64 / (steps.len() - 1) as f64
        } else {
            0.0
        };
        let swell = env(0.08, length * 0.3, length * 0.35, length);
        let tone = Tone::new(penta(s), swell, &sines)
            .vibrato(4.5, 6.0, 0.3)
            .amp(amp);
        b.tone(at, pan, &tone);
    }
}

fn whole(rate: u32) -> Buf {
    let mut b = Buf::new(rate, 1.6);
    grade(&mut b, Verdict::Whole, 0.14);
    for (i, s) in [5, 7].into_iter().enumerate() {
        let chime = Tone::new(penta(s), env(0.0015, 0.0, 0.35, 0.9), &BELL).amp(0.6);
        b.tone(0.08 + 0.16 * i as f64, -0.2 + 0.4 * i as f64, &chime);
    }
    pad(&mut b, 0.1, &[0, 2, 4], 1.2, 0.18);
    b.room(0.3, 0.9);
    b
}

fn snooze(rate: u32) -> Buf {
    let mut b = Buf::new(rate, 2.6);
    grade(&mut b, Verdict::Silent, 0.1);
    // A music box, softly: "Si-lent night" (G A G E), after the stamp.
    let from = STAMP_DOWN + 0.25;
    let tune = [
        (0.0, 783.99, 0.5),
        (0.36, 880.0, 0.3),
        (0.48, 783.99, 0.5),
        (0.72, 659.26, 1.2),
    ];
    for (at, f, length) in tune {
        let tine = Tone::new(f, env(0.001, 0.0, length * 0.45, length), &BELL).amp(0.55);
        b.tone(from + at, 0.0, &tine);
        let octave = Tone::new(2.0 * f, env(0.001, 0.0, 0.12, 0.3), &GLASS).amp(0.15);
        b.tone(from + at, 0.0, &octave);
    }
    pad(&mut b, from + 0.7, &[-5, -3], 1.3, 0.12);
    b.room(0.35, 1.0);
    b
}

/// A rubber stamp from `at`: a rush of air as it comes down
/// ([`STAMP_DOWN`]), then the thunk of it landing, the rubber's slap and
/// the paper flapping under it.
pub fn rubber_stamp(b: &mut Buf, at: f64) {
    let hit = at + STAMP_DOWN;
    let rush = burst(
        (600.0, 2_400.0),
        1.0,
        (STAMP_DOWN * 0.9, 0.01, STAMP_DOWN),
        0.25,
    );
    b.noise(at, 0.0, &rush);
    let sine = [Partial::new(1.0, 1.0, 1.0)];
    let thunk = Tone::new(110.0, env(0.001, 0.015, 0.07, 0.25), &sine).glide(0.5, 0.04);
    b.tone(hit, 0.0, &thunk);
    b.noise(
        hit,
        0.0,
        &burst((900.0, 700.0), 1.8, (0.0004, 0.015, 0.06), 1.0),
    );
    b.noise(
        hit,
        0.0,
        &burst((1_800.0, 700.0), 1.0, (0.0004, 0.025, 0.08), 0.7),
    );
    for (k, (dt, pan)) in [(0.03, -0.3), (0.075, 0.3), (0.12, 0.0)]
        .into_iter()
        .enumerate()
    {
        let flap = burst(
            (2_800.0, 2_000.0),
            1.2,
            (0.002, 0.015, 0.05),
            0.35 / (1.0 + k as f64),
        );
        b.noise(hit + dt, pan, &flap);
    }
}

/// How fast the signature's pen goes `a` seconds after the end, 0 to 1
/// (its fastest): along its stroke ([`signature_point`]) as the stage
/// writes it ([`signed`]), at 1 ms steps.
pub fn signature_speed() -> Vec<f64> {
    let step = 0.001;
    let n = (SIGNING / step).ceil() as usize;
    let at = |a: f64| {
        let (x, y) = signature_point(signed(a) as f32);
        (f64::from(x), f64::from(y))
    };
    let raw: Vec<f64> = (0..n)
        .map(|k| {
            let a = k as f64 * step;
            let (p, q) = (at(a), at(a + step));
            (q.0 - p.0).hypot(q.1 - p.1) / step
        })
        .collect();
    let most = raw.iter().copied().fold(1e-9, f64::max);
    raw.into_iter().map(|v| v / most).collect()
}

fn signature(rate: u32) -> Buf {
    let mut b = Buf::new(rate, SIGNING + 0.3);
    let speed = signature_speed();
    let at = |t: f64| {
        let k = (t / 0.001) as usize;
        speed.get(k).map_or(0.0, |v| v.powf(0.6))
    };
    scratch(&mut b, (0.0, SIGNING), -0.1, 0x516_u64, at);
    // A flick as the pen comes off, a glint on it.
    let flick = burst((5_200.0, 3_500.0), 2.0, (0.0003, 0.004, 0.02), 1.6);
    b.noise(SIGNING - 0.02, 0.0, &flick);
    let glint = Tone::new(penta(12), env(0.001, 0.0, 0.05, 0.18), &GLASS).amp(0.12);
    b.tone(SIGNING - 0.015, 0.0, &glint);
    b
}

fn date(rate: u32) -> Buf {
    let mut b = Buf::new(rate, DATE_FROM + DATING + 0.15);
    handwriting(&mut b, (DATE_FROM, DATING), DATE_SAMPLE, (0.0, 0xDA7E));
    tap(&mut b, DATE_FROM + DATING + 0.02, 1.2);
    b
}

#[cfg(test)]
mod tests {
    use super::super::pen::loudness;
    use super::*;

    /// Each in its place among them, under a name of its own.
    #[test]
    fn every_one_has_its_place() {
        for (k, d) in Diploma::ALL.into_iter().enumerate() {
            assert_eq!(d.index(), k, "{d:?}");
        }
        let names: std::collections::HashSet<String> =
            Diploma::ALL.iter().map(|d| d.name()).collect();
        assert_eq!(names.len(), Diploma::ALL.len());
    }

    /// The signature's scratch follows its stroke: loud where the pen goes
    /// fast, quiet where it slows (its loops' turns, its swash coming to a
    /// stop), silent once it is off but for the flick.
    #[test]
    fn the_signature_follows_its_stroke() {
        let b = signature(48_000);
        let speed = signature_speed();
        let heard = loudness(&b, 0.0, SIGNING - 0.03);
        let pace: Vec<f64> = (0..heard.len())
            .map(|k| {
                let w = &speed[(k * 10).min(speed.len() - 1)..((k + 1) * 10).min(speed.len())];
                w.iter().sum::<f64>() / w.len().max(1) as f64
            })
            .collect();
        let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
        let (mh, mp) = (mean(&heard), mean(&pace));
        let cov: f64 = heard
            .iter()
            .zip(&pace)
            .map(|(h, p)| (h - mh) * (p - mp))
            .sum();
        let var = |x: &[f64], m: f64| x.iter().map(|v| (v - m) * (v - m)).sum::<f64>();
        let r = cov / (var(&heard, mh) * var(&pace, mp)).sqrt();
        assert!(r > 0.7, "the scratch follows the pen's speed: {r}");
        // Quick at first (its loops), slowing to a stop.
        let early = mean(&heard[..20]);
        let late = mean(&heard[heard.len() - 15..]);
        assert!(early > 4.0 * late, "{early} {late}");
        // The pen off with a flick: its glint rings out, then nothing.
        let flick = loudness(&b, SIGNING - 0.02, SIGNING + 0.01);
        let loudest = flick.iter().copied().fold(0.0, f64::max);
        assert!(loudest > 2.0 * late, "{flick:?}");
        let after = loudness(&b, SIGNING + 0.17, SIGNING + 0.3);
        assert!(after.iter().all(|&v| v < 0.02 * early), "{after:?}");
    }

    /// The sheet's rustle follows its front: loudest as it starts to
    /// unroll, quieter as it slows, the front's way ([`unrolled`]).
    #[test]
    fn the_rustle_follows_the_sheet() {
        let b = unroll(48_000);
        let heard = loudness(&b, 0.0, UNROLL * 0.8);
        let early = heard[1..6].iter().sum::<f64>() / 5.0;
        let late = heard[heard.len() - 5..].iter().sum::<f64>() / 5.0;
        assert!(early > 3.0 * late, "{early} {late}");
    }

    /// A rubber stamp lands when the stage brings it down: its loudest
    /// moment just after [`STAMP_DOWN`].
    #[test]
    fn a_stamp_lands_on_the_sheet_when_drawn() {
        let mut b = Buf::new(48_000, 0.6);
        rubber_stamp(&mut b, 0.0);
        let heard = loudness(&b, 0.0, 0.6);
        let loudest = (0..heard.len())
            .max_by(|&a, &z| heard[a].total_cmp(&heard[z]))
            .unwrap();
        let at = loudest as f64 * 0.01;
        assert!((STAMP_DOWN..STAMP_DOWN + 0.04).contains(&at), "{at}");
    }
}
