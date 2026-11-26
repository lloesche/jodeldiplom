//! The diploma's sheet's own sounds ([`Sheet`]), each a score of the
//! palette's atoms written into the synth at once, timed inside to what
//! the stage draws from its beat (the reveal's times and motions, shared
//! with the stage: [`crate::core::singing::Reveal`]'s, [`unrolled`] and
//! [`signed`]). They are on fixed notes (the C major pentatonic on C5:
//! [`penta`]), whatever the song's key, and each sounds the same every
//! time: its notes' seeds start from its own ([`Sheet::seed`]).
//!
//! - [`Sheet::Unroll`]: the sheet unrolling: a soft rustle of paper, as
//!   loud as the sheet's front is fast (quick, then slowing), crinkles in
//!   it, and a soft flump as it lies flat.
//! - [`Sheet::Star`]: a star by the heading popping on: a little glassy
//!   ping, the right one higher.
//! - [`Sheet::Title`]: the song's title landing: a bright "ta-da" in bells
//!   and plucks, glitter off its ends as they sparkle.
//! - [`Sheet::Written`]: an entry written in by hand (the words hit, the
//!   best streak): the pen scratching over its writing time, lifted
//!   between words, then a round "pop" as its value lands, higher for each
//!   entry down the sheet. [`Sheet::Scribbled`]: the pen alone, for an
//!   entry with nothing to cheer for.
//! - [`Sheet::Sticker`]: a star sticker pressed on where no seal goes: a
//!   sticky "tsk", a plop and a glint.
//! - The verdicts without a medal, the grade written in under each:
//!   [`Sheet::Cheer`] a friendly little jingle rising to a chord,
//!   [`Sheet::Whole`] two calm chimes over a soft chord, and
//!   [`Sheet::Snooze`], where nothing was scored, a music box playing
//!   "Silent night" softly, sleepily.
//! - [`Sheet::Medal`]: the medal's wax seal pressed down (a thump, the
//!   wax's soft squelch and a crack), its grade written in, then its
//!   stinger: Bronze a warm plucked chord, Silver bells ringing up to a
//!   chord, Gold a "ta-da-daa" in brass over bells and a cascade of
//!   glitter.
//! - [`Sheet::Stamp`]: a rubber stamp slapped on: a rush of air as it
//!   comes down, the thunk of it landing, the paper flapping under it; and
//!   [`Sheet::NewBest`]: a new best's stamp, then a fanfare, four brass
//!   notes rising to a held chord, a roll of noise under them and glitter
//!   over it.
//! - [`Sheet::Signature`]: the examiner signing: a pen scratching along
//!   the signature's very stroke ([`super::pen::stroke_speed`], at its
//!   pace), louder and higher where it goes fast, a flick as the pen comes
//!   off.
//! - [`Sheet::Date`]: the date written in after it, a tap of the pen on
//!   its end; [`Sheet::Tap`]: the best line shown calmly, a soft tap of
//!   the pen.
//!
//! Each is set to its level ([`Sheet::level`]): the paper's own sounds the
//! softest, the bigger moments a little louder (a verdict without a medal
//! under every medal's); the synth's bus lifts them as it does the game's
//! ([`super::LEVEL`]).

use super::palette::Atom;
use super::pen::{fastest, handwriting, scratch, signature_speed, tap, DATE_SAMPLE};
use super::sounds::{Note, Score};
use crate::core::singing::{
    unrolled, Medal, Side, Verdict, DATE_FROM, DATING, GRADE_WRITE, SIGNING, STAMP_DOWN,
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
    /// written there (its words part where the pen lifts), and the step of
    /// the scale its value lands on (a step up down the sheet).
    fn writing(self) -> (f64, &'static str, i32) {
        match self {
            Entry::Words => (WORDS_FILL, "40 of 45", 2),
            Entry::Streak => (STREAK_WRITE, "17 in a row", 4),
        }
    }
}

/// A sound of the diploma's sheet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sheet {
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
    /// The medal's seal pressed down, and its stinger.
    Medal(Medal),
    /// A rubber stamp slapped on.
    Stamp,
    /// A new best: its stamp and its fanfare.
    NewBest,
    /// The examiner signing.
    Signature,
    /// The date written in.
    Date,
    /// A tap of the pen.
    Tap,
}

impl Sheet {
    /// Every one.
    pub const ALL: [Sheet; 20] = [
        Sheet::Unroll,
        Sheet::Star(Side::Left),
        Sheet::Star(Side::Right),
        Sheet::Title,
        Sheet::Written(Entry::Words),
        Sheet::Written(Entry::Streak),
        Sheet::Scribbled(Entry::Words),
        Sheet::Scribbled(Entry::Streak),
        Sheet::Sticker,
        Sheet::Cheer,
        Sheet::Whole,
        Sheet::Snooze,
        Sheet::Medal(Medal::Bronze),
        Sheet::Medal(Medal::Silver),
        Sheet::Medal(Medal::Gold),
        Sheet::Stamp,
        Sheet::NewBest,
        Sheet::Signature,
        Sheet::Date,
        Sheet::Tap,
    ];

    /// Its name, for its file when the sounds are written out to listen
    /// to.
    pub fn name(self) -> String {
        let entry = |e: Entry| match e {
            Entry::Words => "words",
            Entry::Streak => "streak",
        };
        match self {
            Sheet::Unroll => "diploma-unroll".into(),
            Sheet::Star(Side::Left) => "diploma-star-left".into(),
            Sheet::Star(Side::Right) => "diploma-star-right".into(),
            Sheet::Title => "diploma-title".into(),
            Sheet::Written(e) => format!("diploma-written-{}", entry(e)),
            Sheet::Scribbled(e) => format!("diploma-scribbled-{}", entry(e)),
            Sheet::Sticker => "diploma-sticker".into(),
            Sheet::Cheer => "diploma-cheer".into(),
            Sheet::Whole => "diploma-whole".into(),
            Sheet::Snooze => "diploma-snooze".into(),
            Sheet::Medal(m) => format!("medal-{}", m.name().to_lowercase()),
            Sheet::Stamp => "diploma-stamp".into(),
            Sheet::NewBest => "new-best".into(),
            Sheet::Signature => "diploma-signature".into(),
            Sheet::Date => "diploma-date".into(),
            Sheet::Tap => "diploma-tap".into(),
        }
    }

    /// Its level, dB (0: its notes at the velocities it writes them at;
    /// never above, it only turns them down), set by measure so that the
    /// paper's own sounds are the softest, the bigger moments a little
    /// louder, a verdict without a medal under every medal's.
    pub fn level(self) -> f32 {
        match self {
            Sheet::Unroll => -6.7,
            Sheet::Star(_) => -5.2,
            Sheet::Title => -4.4,
            Sheet::Written(_) => -2.0,
            Sheet::Scribbled(_) => -8.6,
            Sheet::Sticker => -0.7,
            Sheet::Cheer => -1.1,
            Sheet::Whole => -2.9,
            Sheet::Snooze => 0.0,
            Sheet::Medal(_) | Sheet::NewBest => 0.0,
            Sheet::Stamp => -5.4,
            Sheet::Signature => -2.7,
            Sheet::Date => -10.0,
            Sheet::Tap => -12.2,
        }
    }

    /// Where its notes' seeds start: its own, so it sounds the same every
    /// time.
    pub fn seed(self) -> u32 {
        let k = Sheet::ALL.iter().position(|&s| s == self).unwrap_or(0) as u32;
        0x51EE_7000 ^ (k + 1).wrapping_mul(0x9E37_79B9)
    }
}

/// `hz` as a MIDI note.
pub(super) fn midi(hz: f64) -> f32 {
    (69.0 + 12.0 * (hz / 440.0).log2()) as f32
}

/// Step `s` of the major pentatonic scale on C5 (0: C5; 5: C6; below 0
/// under it), as a MIDI note.
pub(super) fn penta(s: i32) -> f32 {
    const STEP: [i32; 5] = [0, 2, 4, 7, 9];
    (72 + 12 * s.div_euclid(5) + STEP[s.rem_euclid(5) as usize]) as f32
}

/// A sheet's sound being written into a score: from `at`, its notes'
/// velocities times `gain` and their pans moved by `pan`.
pub(super) struct Paper<'s, 'a> {
    s: &'s mut Score<'a>,
    at: u64,
    gain: f32,
    pan: f32,
}

impl Paper<'_, '_> {
    /// `atom` on `pitch` (MIDI) `t` seconds in, at `velocity`, at `pan`,
    /// held `hold` seconds.
    pub(super) fn play(
        &mut self,
        t: f64,
        atom: Atom,
        pitch: f32,
        velocity: f32,
        pan: f32,
        hold: f64,
    ) {
        let at = self.at + self.s.secs(t);
        let length = self.s.secs(hold).max(1);
        let note = Note::new(at, atom, pitch)
            .vel(velocity * self.gain)
            .pan(pan + self.pan)
            .hold(length);
        self.s.play(note);
    }

    /// A number from the sound's seed, 0 to 1.
    pub(super) fn unit(&mut self) -> f32 {
        self.s.unit()
    }

    /// A pan anywhere in `-width` to `width`.
    fn scatter(&mut self, width: f32) -> f32 {
        (2.0 * self.unit() - 1.0) * width
    }
}

/// `sound` from `at`, at `gain` (1: at its level), panned by `pan`.
pub(super) fn sheet(s: &mut Score, at: u64, sound: Sheet, gain: f32, pan: f32) {
    let gain = gain * 10f32.powf(sound.level().min(0.0) / 20.0);
    let p = &mut Paper { s, at, gain, pan };
    match sound {
        Sheet::Unroll => unroll(p),
        Sheet::Star(side) => star(p, side),
        Sheet::Title => title(p),
        Sheet::Written(e) => written(p, e, true),
        Sheet::Scribbled(e) => written(p, e, false),
        Sheet::Sticker => sticker(p),
        Sheet::Cheer => cheer(p),
        Sheet::Whole => whole(p),
        Sheet::Snooze => snooze(p),
        Sheet::Medal(m) => medal(p, m),
        Sheet::Stamp => rubber_stamp(p, 0.0),
        Sheet::NewBest => new_best(p),
        Sheet::Signature => signature(p),
        Sheet::Date => {
            handwriting(p, (DATE_FROM, DATING), DATE_SAMPLE, (1.0, 0xDA7E));
            tap(p, DATE_FROM + DATING + 0.02, 0.5);
        }
        Sheet::Tap => tap(p, 0.0, 1.0),
    }
}

/// How fast the sheet's front moves at `t` seconds from its beat, 0 to 1
/// (its fastest, as it starts): as the stage unrolls it ([`unrolled`]).
fn front(t: f64) -> f64 {
    let h = 0.001;
    let pace = |t: f64| (unrolled(t + h) - unrolled((t - h).max(0.0))) / (t + h - (t - h).max(0.0));
    pace(t) / pace(0.0)
}

fn unroll(p: &mut Paper) {
    // The rustle and the air under it, as loud as the front is fast (its
    // envelope falls as the front slows).
    p.play(0.0, Atom::Rustle, midi(2_700.0), 1.0, 0.0, UNROLL);
    p.play(0.0, Atom::Air, midi(320.0), 0.2, 0.0, UNROLL);
    // Crinkles of the paper as it goes, more where it is quicker.
    for _ in 0..16 {
        let at = UNROLL * 0.85 * f64::from(p.unit()).powf(1.6);
        let f = 3_000.0 + 3_000.0 * f64::from(p.unit());
        let amp = 0.8 * front(at) + 0.15;
        let pan = p.scatter(0.4);
        p.play(at, Atom::Snap, midi(f * 0.8), amp as f32, pan, 0.008);
    }
    // It lies flat: a soft flump of air, the paper's edge settling.
    p.play(UNROLL * 0.9, Atom::Knock, midi(84.0), 0.35, 0.0, 0.12);
    p.play(UNROLL * 0.9, Atom::Crack, midi(900.0), 0.5, 0.0, 0.09);
}

fn star(p: &mut Paper, side: Side) {
    let s = match side {
        Side::Left => 8,
        Side::Right => 10,
    };
    p.play(0.0, Atom::Ping, penta(s), 1.0, 0.0, 0.35);
    p.play(0.03, Atom::Glass, penta(s + 5), 0.35, 0.0, 0.2);
}

fn title(p: &mut Paper) {
    // "Ta": a short G; "da!": a C major chord, ringing.
    p.play(0.0, Atom::Plucked, penta(3), 0.7, 0.0, 0.14);
    p.play(0.0, Atom::Ping, penta(8), 0.25, 0.0, 0.14);
    for (i, s) in [5, 7, 8].into_iter().enumerate() {
        let pan = -0.3 + 0.3 * i as f32;
        p.play(0.11, Atom::Plucked, penta(s), 0.45, pan, 0.8);
        p.play(0.11, Atom::Bell, penta(s) + 12.0, 0.2, -pan, 0.6);
    }
    // Glitter off its two ends as they sparkle (0.3 s after it lands).
    for g in 0..8 {
        let side = if g % 2 == 0 { -0.65 } else { 0.65 };
        let at = 0.3 + 0.03 * f64::from(g) + 0.01 * f64::from(p.unit());
        p.play(at, Atom::Glass, penta(11 + g % 4), 0.3, side, 0.22);
    }
}

fn written(p: &mut Paper, entry: Entry, lands: bool) {
    let (length, sample, step) = entry.writing();
    let pen = if lands { 0.1 } else { 1.0 };
    handwriting(p, (0.0, length), sample, (pen, 0x3417 + step as u32));
    if lands {
        // The value in: a round pop rising onto its note, a soft thud of
        // the paper under it, a tock of the nib.
        let f = penta(step);
        p.play(length, Atom::Pop, f, 1.0, 0.0, 0.25);
        p.play(length, Atom::Knock, f - 12.0, 0.1, 0.0, 0.09);
        p.play(length, Atom::Snap, midi(1_100.0), 0.6, 0.0, 0.015);
    }
}

fn sticker(p: &mut Paper) {
    // Peeled off and pressed on: a sticky "tsk", a round plop dropping a
    // little as it settles, then a glint.
    p.play(0.0, Atom::Snap, midi(1_500.0), 0.6, 0.0, 0.02);
    p.play(0.004, Atom::Knock, midi(294.0), 1.0, 0.0, 0.14);
    p.play(0.13, Atom::Glass, penta(10), 0.3, 0.2, 0.3);
}

/// `verdict`'s grade ([`Verdict::grade`]) written in by hand as the stage
/// writes it ([`GRADE_WRITE`]), from the verdict's beat, at `velocity`.
fn grade(p: &mut Paper, verdict: Verdict, velocity: f64) {
    handwriting(p, (0.0, GRADE_WRITE), verdict.grade(), (velocity, 0x6EAD));
}

fn cheer(p: &mut Paper) {
    grade(p, Verdict::Cheer, 0.12);
    // Up a little tune, G C E, to a warm C major chord with a bell on top.
    for (i, s) in [3, 5, 7].into_iter().enumerate() {
        let t = 0.06 + 0.09 * i as f64;
        p.play(
            t,
            Atom::Plucked,
            penta(s),
            0.6,
            -0.25 + 0.25 * i as f32,
            0.3,
        );
    }
    for (i, s) in [5, 7, 8].into_iter().enumerate() {
        p.play(
            0.33,
            Atom::Plucked,
            penta(s),
            0.4,
            -0.3 + 0.3 * i as f32,
            0.9,
        );
    }
    p.play(0.33, Atom::Bell, penta(10), 0.25, 0.0, 0.9);
}

/// A soft chord on `steps`, from `at`, `length` long, at `velocity`.
fn pad(p: &mut Paper, at: f64, steps: &[i32], length: f64, velocity: f32) {
    for (i, &s) in steps.iter().enumerate() {
        let pan = if steps.len() > 1 {
            -0.3 + 0.6 * i as f32 / (steps.len() - 1) as f32
        } else {
            0.0
        };
        p.play(at, Atom::Hum, penta(s), velocity, pan, length);
    }
}

fn whole(p: &mut Paper) {
    grade(p, Verdict::Whole, 0.14);
    for (i, s) in [5, 7].into_iter().enumerate() {
        let t = 0.08 + 0.16 * i as f64;
        p.play(t, Atom::Bell, penta(s), 0.6, -0.2 + 0.4 * i as f32, 0.9);
    }
    pad(p, 0.1, &[0, 2, 4], 1.2, 0.18);
}

fn snooze(p: &mut Paper) {
    grade(p, Verdict::Silent, 0.1);
    // A music box, softly: "Si-lent night" (G A G E), after the stamp.
    let from = STAMP_DOWN + 0.25;
    let tune = [
        (0.0, 79.0, 0.5),
        (0.36, 81.0, 0.3),
        (0.48, 79.0, 0.5),
        (0.72, 76.0, 1.2),
    ];
    for (at, note, length) in tune {
        p.play(from + at, Atom::Bell, note, 0.55, 0.0, length);
        p.play(from + at, Atom::Glass, note + 12.0, 0.15, 0.0, 0.3);
    }
    pad(p, from + 0.7, &[-5, -3], 1.3, 0.12);
}

/// A rubber stamp from `t` seconds: a rush of air as it comes down
/// ([`STAMP_DOWN`]), then the thunk of it landing, the rubber's slap and
/// the paper flapping under it.
fn rubber_stamp(p: &mut Paper, t: f64) {
    let hit = t + STAMP_DOWN;
    p.play(t, Atom::Rush, midi(2_400.0), 0.25, 0.0, STAMP_DOWN);
    p.play(hit, Atom::Thump, midi(55.0), 1.0, 0.0, 0.25);
    p.play(hit, Atom::Crack, midi(700.0), 0.5, 0.0, 0.06);
    p.play(hit, Atom::Crack, midi(1_100.0), 0.35, 0.0, 0.08);
    for (k, (dt, pan)) in [(0.03, -0.3), (0.075, 0.3), (0.12, 0.0)]
        .into_iter()
        .enumerate()
    {
        let flap = 0.35 / (1.0 + k as f32);
        p.play(hit + dt, Atom::Crack, midi(2_000.0), flap, pan, 0.05);
    }
}

/// `medal` stamped down, its grade written in, and its stinger.
fn medal(p: &mut Paper, medal: Medal) {
    // The wax seal pressed down: a thump sliding down, the soft squelch
    // of the wax giving under it, and a crack.
    p.play(0.0, Atom::Thump, midi(56.0), 0.4, 0.0, 0.25);
    p.play(0.0, Atom::Crack, midi(600.0), 0.6, 0.0, 0.12);
    p.play(0.004, Atom::Crack, midi(230.0), 1.0, 0.0, 0.2);
    grade(p, Verdict::Medal(medal), 0.08);
    let at = 0.12;
    match medal {
        Medal::Bronze => {
            for (i, s) in [0, 2, 4, 5].into_iter().enumerate() {
                let t = at + 0.03 * i as f64;
                p.play(t, Atom::Plucked, penta(s), 0.65, -0.3 + 0.2 * i as f32, 1.0);
            }
        }
        Medal::Silver => {
            for (i, s) in [3, 5, 7].into_iter().enumerate() {
                let t = at + 0.09 * i as f64;
                p.play(t, Atom::Bell, penta(s), 0.8, -0.3 + 0.3 * i as f32, 1.0);
            }
            for (i, s) in [5, 7, 8, 10].into_iter().enumerate() {
                let pan = -0.45 + 0.3 * i as f32;
                p.play(at + 0.3, Atom::Plucked, penta(s), 0.56, pan, 1.2);
            }
        }
        Medal::Gold => {
            for (i, f) in [67.0, 71.0, 74.0].into_iter().enumerate() {
                brass(p, at, f, 1.0, -0.3 + 0.3 * i as f32, 0.13);
            }
            for (i, f) in [72.0, 76.0, 79.0, 84.0].into_iter().enumerate() {
                let pan = -0.45 + 0.3 * i as f32;
                brass(p, at + 0.17, f, 1.0, pan, 1.3);
                p.play(at + 0.17, Atom::Bell, f + 12.0, 0.2, -pan, 1.0);
            }
            for g in 0..10 {
                let t = at + 0.25 + 0.045 * f64::from(g);
                let pan = p.scatter(0.8);
                p.play(t, Atom::Glass, penta(9 + g), 0.3, pan, 0.3);
            }
        }
    }
}

/// A brass note on `note` from `t`, at `velocity` and `pan`, held `hold`
/// seconds: both of the brass's layers.
fn brass(p: &mut Paper, t: f64, note: f32, velocity: f32, pan: f32, hold: f64) {
    for atom in super::palette::BRASS {
        p.play(t, atom, note, velocity, pan, hold);
    }
}

/// A new best: its stamp slapped on, then its fanfare.
fn new_best(p: &mut Paper) {
    rubber_stamp(p, 0.0);
    let from = STAMP_DOWN + 0.12;
    // A roll of noise under it, swelling.
    p.play(from, Atom::Roll, midi(1_200.0), 0.25, 0.0, 0.5);
    // "Da da da daaa": G, C, E, then a C major chord, held.
    for (i, f) in [67.0, 72.0, 76.0].into_iter().enumerate() {
        let t = from + 0.12 * i as f64;
        brass(p, t, f, 1.0, 0.0, 0.11);
        brass(p, t, f - 12.0, 0.5, 0.0, 0.11);
    }
    for (i, f) in [72.0, 76.0, 79.0, 84.0, 88.0].into_iter().enumerate() {
        brass(p, from + 0.36, f, 0.75, -0.5 + 0.25 * i as f32, 1.5);
    }
    for g in 0..12 {
        let t = from + 0.4 + 0.06 * f64::from(g);
        let pan = p.scatter(0.8);
        p.play(t, Atom::Glass, penta(10 + g % 6), 0.2, pan, 0.3);
    }
}

fn signature(p: &mut Paper) {
    let most = fastest();
    scratch(p, (0.0, SIGNING), -0.1, |t| {
        signature_speed(t - 0.5 * super::pen::GRAIN, most).powf(0.6)
    });
    // A flick as the pen comes off, a glint on it.
    p.play(SIGNING - 0.02, Atom::Snap, midi(3_500.0), 1.0, 0.0, 0.02);
    p.play(SIGNING - 0.015, Atom::Glass, penta(12), 0.12, 0.0, 0.18);
}
