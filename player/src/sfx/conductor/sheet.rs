//! The diploma's sheet's own sounds (the results screen's foley,
//! [`Sheet`]), each a score of the
//! palette's atoms written into the synth at once, timed inside to what
//! the stage draws from its beat (the reveal's times and motions, shared
//! with the stage: [`crate::core::karaoke::result::Reveal`]'s, and its
//! paces, [`unrolling`] and [`crate::core::karaoke::result::signing`]).
//! They are on fixed notes (the C major pentatonic on C5:
//! [`penta`]), whatever the song's key, and each sounds the same every
//! time: its notes' seeds start from its own ([`Sheet::seed`]). Its notes
//! are written in seconds from where it plays, at its level and its gain,
//! panned by where on the sheet it lands (the score's origin).
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
//!   the signature's very stroke ([`super::pen::signature_speed`], at its
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

use jodel_synth::dsp;

use super::palette::{Atom, BRASS};
use super::pen::{handwriting, scratch, signature_speed, tap, DATE_SAMPLE};
use super::sounds::{Note, Score};
use crate::core::karaoke::result::{
    unrolling, Medal, Side, Verdict, DATE_FROM, DATING, GRADE_WRITE, SIGNING, STAMP_DOWN,
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

/// Every sheet's sound in its order, with its name (its file's, when the
/// sounds are written out to listen to) and its title (as the Synth tab
/// lists it).
const SHEETS: [(Sheet, &str, &str); 20] = [
    (Sheet::Unroll, "diploma-unroll", "Unrolled"),
    (Sheet::Star(Side::Left), "diploma-star-left", "A star, left"),
    (
        Sheet::Star(Side::Right),
        "diploma-star-right",
        "A star, right",
    ),
    (Sheet::Title, "diploma-title", "The title"),
    (
        Sheet::Written(Entry::Words),
        "diploma-written-words",
        "Written: the words",
    ),
    (
        Sheet::Written(Entry::Streak),
        "diploma-written-streak",
        "Written: the streak",
    ),
    (
        Sheet::Scribbled(Entry::Words),
        "diploma-scribbled-words",
        "Scribbled: the words",
    ),
    (
        Sheet::Scribbled(Entry::Streak),
        "diploma-scribbled-streak",
        "Scribbled: the streak",
    ),
    (Sheet::Sticker, "diploma-sticker", "A sticker"),
    (Sheet::Cheer, "diploma-cheer", "A cheer"),
    (Sheet::Whole, "diploma-whole", "Not from the start"),
    (Sheet::Snooze, "diploma-snooze", "Nothing scored"),
    (Sheet::Medal(Medal::Bronze), "medal-bronze", "Medal: Bronze"),
    (Sheet::Medal(Medal::Silver), "medal-silver", "Medal: Silver"),
    (Sheet::Medal(Medal::Gold), "medal-gold", "Medal: Gold"),
    (Sheet::Stamp, "diploma-stamp", "The stamp"),
    (Sheet::NewBest, "new-best", "A new best"),
    (Sheet::Signature, "diploma-signature", "Signed"),
    (Sheet::Date, "diploma-date", "Dated"),
    (Sheet::Tap, "diploma-tap", "A tap of the pen"),
];

impl Sheet {
    /// Every one.
    pub const ALL: [Sheet; SHEETS.len()] = {
        let mut all = [Sheet::Unroll; SHEETS.len()];
        let mut k = 0;
        while k < all.len() {
            all[k] = SHEETS[k].0;
            k += 1;
        }
        all
    };

    /// Its row of [`SHEETS`].
    fn row(self) -> (Sheet, &'static str, &'static str) {
        let row = SHEETS.iter().find(|row| row.0 == self);
        *row.expect("every sheet's sound has its row")
    }

    /// Its name, for its file when the sounds are written out to listen
    /// to.
    pub fn name(self) -> &'static str {
        self.row().1
    }

    /// Its title, as the Synth tab lists it.
    pub fn title(self) -> &'static str {
        self.row().2
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

/// Step `s` of the major pentatonic scale on C5 (0: C5; 5: C6; below 0
/// under it), as a MIDI note.
pub(super) fn penta(s: i32) -> f32 {
    const STEP: [i32; 5] = [0, 2, 4, 7, 9];
    (72 + 12 * s.div_euclid(5) + STEP[s.rem_euclid(5) as usize]) as f32
}

/// The steps of a strum or a run, each with its time and its pan: the
/// first from `t0` at `pan0`, each next `dt` later and `dpan` further.
fn strum<T>(
    steps: impl IntoIterator<Item = T>,
    (t0, dt): (f64, f64),
    (pan0, dpan): (f32, f32),
) -> impl Iterator<Item = (f64, T, f32)> {
    (steps.into_iter().enumerate())
        .map(move |(i, s)| (t0 + dt * i as f64, s, pan0 + dpan * i as f32))
}

/// `sound` into `p` (its origin where it plays, at its level and gain).
pub(super) fn sheet(p: &mut Score, sound: Sheet) {
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
        Sheet::Stamp => rubber_stamp(p, 1.0),
        Sheet::NewBest => new_best(p),
        Sheet::Signature => signature(p),
        Sheet::Date => {
            handwriting(p, (DATE_FROM, DATING), DATE_SAMPLE, 1.0);
            tap(p, DATE_FROM + DATING + 0.02, 0.5);
        }
        Sheet::Tap => tap(p, 0.0, 1.0),
    }
}

/// How fast the sheet's front moves at `t` seconds from its beat, 0 to 1
/// (its fastest, as it starts): as the stage unrolls it ([`unrolling`]).
fn front(t: f64) -> f64 {
    unrolling(t) / unrolling(0.0)
}

fn unroll(p: &mut Score) {
    // The rustle and the air under it, as loud as the front is fast (its
    // envelope falls as the front slows), its band rising as the sheet
    // goes.
    let rustle = Note::at_secs(0.0, Atom::Rustle, dsp::note(2_700.0));
    p.play(rustle.hold_secs(UNROLL).glide(UNROLL));
    p.play(
        Note::at_secs(0.0, Atom::Air, dsp::note(320.0))
            .vel(0.2)
            .hold_secs(UNROLL),
    );
    // Crinkles of the paper as it goes, more where it is quicker.
    for _ in 0..16 {
        let at = UNROLL * 0.85 * f64::from(p.unit()).powf(1.6);
        let f = 3_000.0 + 3_000.0 * f64::from(p.unit());
        let amp = 0.8 * front(at) + 0.15;
        let pan = p.scatter(0.4);
        p.play(
            Note::at_secs(at, Atom::Snap, dsp::note(f * 0.8))
                .vel(amp as f32)
                .pan(pan)
                .hold_secs(0.008),
        );
    }
    // It lies flat: a soft flump of air, the paper's edge settling.
    p.play(
        Note::at_secs(UNROLL * 0.9, Atom::Knock, dsp::note(84.0))
            .vel(0.35)
            .hold_secs(0.12),
    );
    p.play(
        Note::at_secs(UNROLL * 0.9, Atom::Crack, dsp::note(900.0))
            .vel(0.5)
            .hold_secs(0.09),
    );
}

fn star(p: &mut Score, side: Side) {
    let s = match side {
        Side::Left => 8,
        Side::Right => 10,
    };
    p.play(Note::at_secs(0.0, Atom::Ping, penta(s)).hold_secs(0.35));
    p.play(
        Note::at_secs(0.03, Atom::Glass, penta(s + 5))
            .vel(0.35)
            .hold_secs(0.2),
    );
}

fn title(p: &mut Score) {
    // "Ta": a short G; "da!": a C major chord, ringing.
    p.play(
        Note::at_secs(0.0, Atom::Plucked, penta(3))
            .vel(0.7)
            .hold_secs(0.14),
    );
    p.play(
        Note::at_secs(0.0, Atom::Ping, penta(8))
            .vel(0.25)
            .hold_secs(0.14),
    );
    for (t, s, pan) in strum([5, 7, 8], (0.11, 0.0), (-0.3, 0.3)) {
        p.play(
            Note::at_secs(t, Atom::Plucked, penta(s))
                .vel(0.45)
                .pan(pan)
                .hold_secs(0.8),
        );
        p.play(
            Note::at_secs(t, Atom::Bell, penta(s) + 12.0)
                .vel(0.2)
                .pan(-pan)
                .hold_secs(0.6),
        );
    }
    // Glitter off its two ends as they sparkle (0.3 s after it lands).
    for g in 0..8 {
        let side = if g % 2 == 0 { -0.65 } else { 0.65 };
        let at = 0.3 + 0.03 * f64::from(g) + 0.01 * f64::from(p.unit());
        p.play(
            Note::at_secs(at, Atom::Glass, penta(11 + g % 4))
                .vel(0.3)
                .pan(side)
                .hold_secs(0.22),
        );
    }
}

fn written(p: &mut Score, entry: Entry, lands: bool) {
    let (length, sample, step) = entry.writing();
    let pen = if lands { 0.1 } else { 1.0 };
    handwriting(p, (0.0, length), sample, pen);
    if lands {
        // The value in: a round pop rising onto its note, a soft thud of
        // the paper under it, a tock of the nib.
        let f = penta(step);
        p.play(Note::at_secs(length, Atom::Pop, f).hold_secs(0.25));
        p.play(
            Note::at_secs(length, Atom::Knock, f - 12.0)
                .vel(0.1)
                .hold_secs(0.09),
        );
        p.play(
            Note::at_secs(length, Atom::Snap, dsp::note(1_100.0))
                .vel(0.6)
                .hold_secs(0.015),
        );
    }
}

fn sticker(p: &mut Score) {
    // Peeled off and pressed on: a sticky "tsk", a round plop dropping a
    // little as it settles, then a glint.
    p.play(
        Note::at_secs(0.0, Atom::Snap, dsp::note(1_500.0))
            .vel(0.6)
            .hold_secs(0.02),
    );
    p.play(Note::at_secs(0.004, Atom::Knock, dsp::note(294.0)).hold_secs(0.14));
    p.play(
        Note::at_secs(0.13, Atom::Glass, penta(10))
            .vel(0.3)
            .pan(0.2)
            .hold_secs(0.3),
    );
}

/// `verdict`'s grade ([`Verdict::grade`]) written in by hand as the stage
/// writes it ([`GRADE_WRITE`]), from the verdict's beat, at `velocity`.
fn grade(p: &mut Score, verdict: Verdict, velocity: f64) {
    handwriting(p, (0.0, GRADE_WRITE), verdict.grade(), velocity);
}

fn cheer(p: &mut Score) {
    grade(p, Verdict::Cheer, 0.12);
    // Up a little tune, G C E, to a warm C major chord with a bell on top.
    for (t, s, pan) in strum([3, 5, 7], (0.06, 0.09), (-0.25, 0.25)) {
        p.play(
            Note::at_secs(t, Atom::Plucked, penta(s))
                .vel(0.6)
                .pan(pan)
                .hold_secs(0.3),
        );
    }
    for (t, s, pan) in strum([5, 7, 8], (0.33, 0.0), (-0.3, 0.3)) {
        p.play(
            Note::at_secs(t, Atom::Plucked, penta(s))
                .vel(0.4)
                .pan(pan)
                .hold_secs(0.9),
        );
    }
    p.play(
        Note::at_secs(0.33, Atom::Bell, penta(10))
            .vel(0.25)
            .hold_secs(0.9),
    );
}

/// A soft chord on `steps`, from `at`, `length` long, at `velocity`.
fn pad(p: &mut Score, at: f64, steps: &[i32], length: f64, velocity: f32) {
    let (from, step) = match steps.len() {
        0 | 1 => (0.0, 0.0),
        n => (-0.3, 0.6 / (n - 1) as f32),
    };
    for (t, &s, pan) in strum(steps, (at, 0.0), (from, step)) {
        p.play(
            Note::at_secs(t, Atom::Hum, penta(s))
                .vel(velocity)
                .pan(pan)
                .hold_secs(length),
        );
    }
}

fn whole(p: &mut Score) {
    grade(p, Verdict::Whole, 0.14);
    for (t, s, pan) in strum([5, 7], (0.08, 0.16), (-0.2, 0.4)) {
        p.play(
            Note::at_secs(t, Atom::Bell, penta(s))
                .vel(0.6)
                .pan(pan)
                .hold_secs(0.9),
        );
    }
    pad(p, 0.1, &[0, 2, 4], 1.2, 0.18);
}

fn snooze(p: &mut Score) {
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
        p.play(
            Note::at_secs(from + at, Atom::Bell, note)
                .vel(0.55)
                .hold_secs(length),
        );
        p.play(
            Note::at_secs(from + at, Atom::Glass, note + 12.0)
                .vel(0.15)
                .hold_secs(0.3),
        );
    }
    pad(p, from + 0.7, &[-5, -3], 1.3, 0.12);
}

/// A rubber stamp: a rush of air as it comes down ([`STAMP_DOWN`]), then
/// the thunk of it landing (its thump at `thump`), the rubber's slap and
/// the paper flapping under it.
fn rubber_stamp(p: &mut Score, thump: f32) {
    let hit = STAMP_DOWN;
    let rush = Note::at_secs(0.0, Atom::Rush, dsp::note(2_400.0)).vel(0.25);
    p.play(rush.hold_secs(STAMP_DOWN).glide(STAMP_DOWN));
    p.play(
        Note::at_secs(hit, Atom::Thump, dsp::note(55.0))
            .vel(thump)
            .hold_secs(0.25),
    );
    p.play(
        Note::at_secs(hit, Atom::Crack, dsp::note(700.0))
            .vel(0.5)
            .hold_secs(0.06),
    );
    p.play(
        Note::at_secs(hit, Atom::Crack, dsp::note(1_100.0))
            .vel(0.35)
            .hold_secs(0.08),
    );
    for (k, (dt, pan)) in [(0.03, -0.3), (0.075, 0.3), (0.12, 0.0)]
        .into_iter()
        .enumerate()
    {
        let flap = 0.35 / (1.0 + k as f32);
        p.play(
            Note::at_secs(hit + dt, Atom::Crack, dsp::note(2_000.0))
                .vel(flap)
                .pan(pan)
                .hold_secs(0.05),
        );
    }
}

/// `medal` stamped down, its grade written in, and its stinger.
fn medal(p: &mut Score, medal: Medal) {
    // The wax seal pressed down: a thump sliding down, the soft squelch
    // of the wax giving under it, and a crack.
    p.play(
        Note::at_secs(0.0, Atom::Thump, dsp::note(56.0))
            .vel(0.4)
            .hold_secs(0.25),
    );
    p.play(
        Note::at_secs(0.0, Atom::Crack, dsp::note(600.0))
            .vel(0.6)
            .hold_secs(0.12),
    );
    p.play(Note::at_secs(0.004, Atom::Crack, dsp::note(230.0)).hold_secs(0.2));
    grade(p, Verdict::Medal(medal), 0.08);
    let at = 0.12;
    match medal {
        Medal::Bronze => {
            for (t, s, pan) in strum([0, 2, 4, 5], (at, 0.03), (-0.3, 0.2)) {
                p.play(
                    Note::at_secs(t, Atom::Plucked, penta(s))
                        .vel(0.65)
                        .pan(pan)
                        .hold_secs(1.0),
                );
            }
        }
        Medal::Silver => {
            for (t, s, pan) in strum([3, 5, 7], (at, 0.09), (-0.3, 0.3)) {
                p.play(
                    Note::at_secs(t, Atom::Bell, penta(s))
                        .vel(0.8)
                        .pan(pan)
                        .hold_secs(1.0),
                );
            }
            for (t, s, pan) in strum([5, 7, 8, 10], (at + 0.3, 0.0), (-0.45, 0.3)) {
                p.play(
                    Note::at_secs(t, Atom::Plucked, penta(s))
                        .vel(0.56)
                        .pan(pan)
                        .hold_secs(1.2),
                );
            }
        }
        Medal::Gold => {
            for (t, f, pan) in strum([67.0, 71.0, 74.0], (at, 0.0), (-0.3, 0.3)) {
                brass(p, t, f, 1.0, pan, 0.13);
            }
            for (t, f, pan) in strum([72.0, 76.0, 79.0, 84.0], (at + 0.17, 0.0), (-0.45, 0.3)) {
                brass(p, t, f, 1.0, pan, 1.3);
                p.play(
                    Note::at_secs(t, Atom::Bell, f + 12.0)
                        .vel(0.2)
                        .pan(-pan)
                        .hold_secs(1.0),
                );
            }
            for g in 0..10 {
                let t = at + 0.25 + 0.045 * f64::from(g);
                let pan = p.scatter(0.8);
                p.play(
                    Note::at_secs(t, Atom::Glass, penta(9 + g))
                        .vel(0.3)
                        .pan(pan)
                        .hold_secs(0.3),
                );
            }
        }
    }
}

/// A brass note on `note` from `t`, at `velocity` and `pan`, held `hold`
/// seconds: both of the brass's layers.
fn brass(p: &mut Score, t: f64, note: f32, velocity: f32, pan: f32, hold: f64) {
    for atom in BRASS {
        p.play(
            Note::at_secs(t, atom, note)
                .vel(velocity)
                .pan(pan)
                .hold_secs(hold),
        );
    }
}

/// How hard a new best's stamp thumps: under the stamp's own, so the
/// fanfare after it keeps its weight over the low end.
const NEW_BEST_THUMP: f32 = 0.5;

/// A new best: its stamp slapped on (its thump softer), then its fanfare.
fn new_best(p: &mut Score) {
    rubber_stamp(p, NEW_BEST_THUMP);
    let from = STAMP_DOWN + 0.12;
    // A roll of noise under it, swelling.
    p.play(
        Note::at_secs(from, Atom::Roll, dsp::note(1_200.0))
            .vel(0.25)
            .hold_secs(0.5),
    );
    // "Da da da daaa": G, C, E, then a C major chord, held.
    for (t, f, _) in strum([67.0, 72.0, 76.0], (from, 0.12), (0.0, 0.0)) {
        brass(p, t, f, 1.0, 0.0, 0.11);
        brass(p, t, f - 12.0, 0.5, 0.0, 0.11);
    }
    let chord = [72.0, 76.0, 79.0, 84.0, 88.0];
    for (t, f, pan) in strum(chord, (from + 0.36, 0.0), (-0.5, 0.25)) {
        brass(p, t, f, 0.75, pan, 1.5);
    }
    for g in 0..12 {
        let t = from + 0.4 + 0.06 * f64::from(g);
        let pan = p.scatter(0.8);
        p.play(
            Note::at_secs(t, Atom::Glass, penta(10 + g % 6))
                .vel(0.2)
                .pan(pan)
                .hold_secs(0.3),
        );
    }
}

fn signature(p: &mut Score) {
    scratch(p, (0.0, SIGNING), -0.1, |grain| {
        signature_speed(grain).powf(0.6)
    });
    // A flick as the pen comes off, a glint on it.
    p.play(Note::at_secs(SIGNING - 0.02, Atom::Snap, dsp::note(3_500.0)).hold_secs(0.02));
    p.play(
        Note::at_secs(SIGNING - 0.015, Atom::Glass, penta(12))
            .vel(0.12)
            .hold_secs(0.18),
    );
}
