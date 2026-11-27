//! The game's sounds put together from the palette's atoms, in the song's
//! key and tempo, as timed notes sent to the synth ([`Score`], [`Note`]).
//! Each sound's layers are its tier's row's ([`super::Layers`]); its size
//! beyond them follows what it celebrates (a `share` of the way to the
//! top, 0 to 1: a rung's of the rungs, a tier's of the tiers): more notes
//! and longer, never only louder. The timing comes from the conductor (the
//! grid's landing rule): `now` is the event's frame, `hit` the frame its
//! main hit lands on, `at` a sound's start.
//!
//! With no key known the pitched layers (plucks, brass, choir, pads, the
//! low tonic, pips and the horn) stay out, and what rings (bells, glitter,
//! coins) sits on fixed notes high over the song; the ladder becomes the
//! pop growing brighter, not higher.
//!
//! - [`pop`]: a word's bubble on the ladder's step; a glint over it at the
//!   rungs that have one. [`clink`]: a coin dropped, a tiny bounce after
//!   it.
//! - [`zap`]: the rescue shot.
//! - [`wind_down`]: the streak lost: from a rung with the reels, the reels
//!   running down (ticks slowing, pips falling, a whirr sinking, a muted
//!   "wah"), longer the higher it was; else two soft notes falling.
//! - [`stinger`]: a rung reached: a pickup, the hit on the tonic, then an
//!   arpeggio up the chord in sixteenths; by its layers the hit an octave
//!   down with a bell, a bell's tremolo in thirty-seconds and the
//!   register's ka-ching, a held brass chord, glitter.
//! - [`chime`]: a phrase rated: bells struck in a quick roll, then by its
//!   tier's layers plucks, a brass hit or a "ta-daa", a choir and glitter.
//! - [`payout`]: the top rung's encore on a downbeat: bells climbing, then
//!   quickening, coins pouring so many a beat, the win on the next bar's
//!   first beat, coins thinning a bar after it.
//! - [`jackpot`]: the top tier's: the slam, a choir and brass, the bell's
//!   tremolo, a coin shower and glitter rising.
//! - Held: [`riser`] into a high rung, let go a breath before its hit
//!   ([`riser_lands`]); the [`shimmer`] pad while the streak is high, each
//!   word holding it on; [`glitter_after`] each word at the top. Each note
//!   of them is held for a bounded time of its own, so a release that
//!   never arrives costs seconds, not the song.
//! - [`tick`], [`gem`]: the diploma's count-up and its gems.
//! - [`found`]: a step of a song's preparation found something: a soft
//!   pluck, a glint an octave over it.

use jodel_synth::{Command, Commands, Cue, NoteOn, Tag};

use super::key::{Pentatonic, DEGREES};
use super::ladder::Step;
use super::layers::{chime_brass, chime_has, tier_share, Chime, Streak};
use super::palette::{self, Atom, BRASS, COINS, SLAM};
use super::tags::{PAYOUT, RISER, SHIMMER, STREAK};
use crate::tiers::Tier;

/// How long the jackpot and the payout are, beats (bars of `bar` beats),
/// and the beat the payout's whoosh leads into it with: what they ask
/// room for between the lines.
pub(crate) const JACKPOT_BEATS: f64 = 3.0;
pub(super) fn payout_beats(bar: u32) -> f64 {
    2.0 * f64::from(bar.max(2))
}
pub(super) const PAYOUT_LEAD: f64 = 1.0;

/// A pickup needs this long before its hit, seconds; the riser falls
/// silent this long before the hit it leads into, and holds this many
/// beats at most (no rung come: it ends by itself); the shimmer holds this
/// many bars from the word that last held it on.
const PICKUP_MIN: f64 = 0.08;
const BREATH: f64 = 0.06;
const RISE_WAIT: f64 = 3.0;
const SHIMMER_BARS: f64 = 2.0;

/// Where each kind of note sits: the lowest its tonic is put at.
const POP_HOME: i32 = 72;
const STING_HOME: i32 = 60;
const BELL_HOME: i32 = 72;
const CLINK_HOME: i32 = 96;
const LOW_HOME: i32 = 36;
const WAH_HOME: i32 = 55;
const PAD_HOME: i32 = 84;

/// What rings where no key is known: high over any song.
const HIGH_NOTES: [f32; 4] = [96.0, 103.0, 108.0, 115.0];

/// The degrees of the key the six clinks ring on, in turn.
const CLINK_DEGREES: [i32; 6] = [0, 2, 4, 1, 3, 5];

/// When a note is: on a frame (a game's), or seconds from the score's
/// origin (a sheet's sound's, [`Origin`]).
#[derive(Clone, Copy, Debug)]
enum Time {
    Frame(u64),
    Secs(f64),
}

/// A note to send: when, its atom, pitch (MIDI), velocity, pan, shape
/// (its own: a sung vowel), length (none: the patch's gate), tag, its
/// glide's time (0: the patch's) and its mono patch's line.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Note {
    at: Time,
    atom: Atom,
    pitch: f32,
    velocity: f32,
    pan: f32,
    shape: f32,
    length: Option<Time>,
    tag: Tag,
    glide: Option<f32>,
    line: u8,
}

impl Note {
    /// `atom` on `pitch` at frame `at`: full, in the middle, as the patch
    /// holds it, a sound of its own.
    pub(crate) fn new(at: u64, atom: Atom, pitch: f32) -> Note {
        Note {
            at: Time::Frame(at),
            atom,
            pitch,
            velocity: 1.0,
            pan: 0.0,
            shape: 0.0,
            length: None,
            tag: Tag::NONE,
            glide: None,
            line: 0,
        }
    }

    /// The same `t` seconds from the score's origin ([`Origin::at`]).
    pub(super) fn at_secs(t: f64, atom: Atom, pitch: f32) -> Note {
        Note {
            at: Time::Secs(t),
            ..Note::new(0, atom, pitch)
        }
    }

    pub(crate) fn vel(self, velocity: f32) -> Note {
        Note { velocity, ..self }
    }

    pub(super) fn pan(self, pan: f32) -> Note {
        Note { pan, ..self }
    }

    pub(super) fn shape(self, shape: f32) -> Note {
        Note { shape, ..self }
    }

    pub(crate) fn hold(self, frames: u64) -> Note {
        Note {
            length: Some(Time::Frame(frames)),
            ..self
        }
    }

    /// Held `seconds` (a frame at least).
    pub(super) fn hold_secs(self, seconds: f64) -> Note {
        Note {
            length: Some(Time::Secs(seconds)),
            ..self
        }
    }

    pub(crate) fn tag(self, tag: Tag) -> Note {
        Note { tag, ..self }
    }

    /// Its pitch's glide taking `seconds` (as long as what it follows).
    pub(super) fn glide(self, seconds: f64) -> Note {
        Note {
            glide: Some(seconds as f32),
            ..self
        }
    }

    /// On line `line` of its mono patch (a voice each).
    fn line(self, line: u8) -> Note {
        Note { line, ..self }
    }
}

/// Where a score's notes are from: frames added to theirs, a gain their
/// velocities are taken by, a pan added to theirs; none of it for the
/// game's own ([`Origin::NONE`]), a sheet's sound's from where it plays.
#[derive(Clone, Copy, Debug)]
pub(super) struct Origin {
    pub(super) at: u64,
    pub(super) gain: f32,
    pub(super) pan: f32,
}

impl Origin {
    /// None: the notes as they are.
    pub(super) const NONE: Origin = Origin {
        at: 0,
        gain: 1.0,
        pan: 0.0,
    };
}

/// What the sounds are written with: where the commands go (and the count
/// of those not taken), the theme whose patches play (its place), the key
/// (if known), frames a beat, beats a bar, frames a second, the
/// variations' seed, and where the notes are from.
pub(crate) struct Score<'a> {
    pub(super) out: &'a mut dyn Commands,
    pub(super) dropped: &'a mut u64,
    pub(super) theme: usize,
    pub(super) key: Option<Pentatonic>,
    pub(super) beat: f64,
    pub(super) bar: u32,
    pub(super) rate: f64,
    pub(super) seed: &'a mut u32,
    pub(super) origin: Origin,
}

impl Score<'_> {
    /// `c` sent, counted if it was not taken; whether it was.
    pub(super) fn send(&mut self, c: Command) -> bool {
        let taken = self.out.send(c);
        if !taken {
            *self.dropped += 1;
        }
        taken
    }

    /// `t` in frames from the origin.
    fn frames(&self, t: Time) -> u64 {
        match t {
            Time::Frame(f) => f,
            Time::Secs(s) => self.secs(s),
        }
    }

    /// `n` sent, its seed the next, from the origin.
    pub(crate) fn play(&mut self, n: Note) {
        let seed = jodel_dsp::xorshift32(self.seed);
        let o = self.origin;
        let length = n.length.map(|l| match l {
            Time::Frame(f) => f,
            Time::Secs(s) => self.secs(s).max(1),
        });
        self.send(Command::At {
            frame: o.at + self.frames(n.at),
            tag: n.tag,
            cue: Cue::Note {
                patch: palette::id(self.theme, n.atom),
                note: NoteOn {
                    note: n.pitch,
                    velocity: (n.velocity * o.gain).clamp(0.0, 1.0),
                    seed,
                    pan: (n.pan + o.pan).clamp(-1.0, 1.0),
                    shape: n.shape.clamp(-1.0, 1.0),
                    glide: n.glide,
                    line: n.line,
                },
                length,
            },
        });
    }

    /// The held voices of `tag` let go at `at` (whatever their patch: the
    /// theme's they were played in).
    pub(crate) fn release(&mut self, at: u64, tag: Tag) {
        self.send(Command::At {
            frame: at,
            tag,
            cue: Cue::Release,
        });
    }

    /// Degree `d` of the pentatonic over the tonic at `floor` (none with
    /// no key).
    fn deg(&self, d: i32, floor: i32) -> Option<f32> {
        self.key.map(|k| k.note(d, k.home(floor)))
    }

    /// Chord tone `k` over the tonic at `floor` (none with no key).
    fn tonal(&self, k: i32, floor: i32) -> Option<f32> {
        self.key.map(|key| key.note(key.chord(k), key.home(floor)))
    }

    /// What rings for chord tone `k`: the chord tone over `floor`, or with
    /// no key one of the high notes.
    fn bright(&self, k: i32, floor: i32) -> f32 {
        let fixed = HIGH_NOTES[k.rem_euclid(HIGH_NOTES.len() as i32) as usize];
        self.tonal(k, floor).unwrap_or(fixed)
    }

    /// `beats` beats, in frames.
    fn beats(&self, beats: f64) -> u64 {
        (beats * self.beat).round().max(0.0) as u64
    }

    /// `seconds` seconds, in frames.
    pub(super) fn secs(&self, seconds: f64) -> u64 {
        (seconds * self.rate).round().max(0.0) as u64
    }

    /// A number from the seed, 0 to 1.
    pub(super) fn unit(&mut self) -> f32 {
        jodel_synth::dsp::unit(self.seed)
    }

    /// The next seed of its own (a sound's part with a stream of its own:
    /// a hand's letters).
    pub(super) fn draw(&mut self) -> u32 {
        jodel_dsp::xorshift32(self.seed)
    }

    /// A pan anywhere in `-width` to `width`.
    pub(super) fn scatter(&mut self, width: f32) -> f32 {
        (2.0 * self.unit() - 1.0) * width
    }
}

/// A word's bubble at `at`, on the ladder's `step`, `share` of the rungs
/// reached (a glint over it where `glinted`), at `pan`, `size` big (0 as
/// any to 1, held full the longest): the bigger, the louder, with a body
/// an octave under it.
pub(super) fn pop(
    s: &mut Score,
    at: u64,
    step: Step,
    share: f32,
    glinted: bool,
    pan: f32,
    size: f32,
) {
    let glint = 0.3 + 0.5 * share;
    let big = size.clamp(0.0, 1.0);
    match s.deg(step.degree, POP_HOME) {
        Some(note) => {
            s.play(Note::new(at, Atom::Pop, note).vel(0.8 + 0.2 * big).pan(pan));
            let body = Note::new(at, Atom::PopBody, note);
            s.play(body.vel(0.7 + 0.3 * big).pan(pan));
            if big > 0.0 {
                let under = Note::new(at, Atom::PopBody, note - 12.0);
                s.play(under.vel(0.6 * big).pan(pan));
            }
            if glinted {
                let over = Note::new(at, Atom::Glint, note + 12.0);
                s.play(over.vel(glint).pan(pan));
            }
        }
        None => {
            // Brighter, not higher.
            let bright = (0.45 + 0.1 * step.degree as f32) * (1.0 + 0.3 * big);
            s.play(Note::new(at, Atom::Pop, HIGH_NOTES[0]).vel(bright).pan(pan));
            let over = Note::new(at, Atom::Glint, HIGH_NOTES[1]);
            s.play(over.vel(glint * bright).pan(pan));
        }
    }
}

/// Clink `k` (taken in turn) at `at`, at `velocity` and `pan`.
pub(super) fn clink(s: &mut Score, at: u64, k: usize, velocity: f32, pan: f32) {
    let atom = COINS[k % COINS.len()];
    let d = CLINK_DEGREES[k % CLINK_DEGREES.len()];
    let note = s.deg(d, CLINK_HOME).unwrap_or(HIGH_NOTES[k % 2]);
    s.play(Note::new(at, atom, note).vel(velocity).pan(pan));
    let bounce = Note::new(at + s.secs(0.03), atom, note + 1.0);
    s.play(bounce.vel(0.3 * velocity).pan(pan));
}

/// The rescue shot at `at`, at `pan` (unpitched: a dive).
pub(super) fn zap(s: &mut Score, at: u64, pan: f32) {
    s.play(Note::new(at, Atom::Zap, 52.0).pan(pan));
}

/// The streak lost at `at`: from a rung with the reels, `reels` of the way
/// from the first such rung to the top (0 to 1), the reels running down,
/// the longer the higher it was; else two soft notes falling.
pub(super) fn wind_down(s: &mut Score, at: u64, reels: Option<f32>) {
    let Some(q) = reels else {
        for (k, d) in [2, 0].into_iter().enumerate() {
            let t = at + s.secs(0.09 * k as f64);
            match s.deg(d, POP_HOME) {
                Some(note) => s.play(Note::new(t, Atom::Pip, note).vel(0.45)),
                None => s.play(Note::new(t, Atom::Tick, 60.0).vel(0.5)),
            }
        }
        return;
    };
    let q = f64::from(q).clamp(0.0, 1.0);
    let length = 0.4 + 0.5 * q;
    let ticks = 6 + (6.0 * q).round() as i32;
    let (mut t, mut pace) = (0.0, 0.025);
    for k in 0..ticks {
        let tick = at + s.secs(t);
        let pan = if k % 2 == 0 { -0.15 } else { 0.15 };
        s.play(Note::new(tick, Atom::Tick, 60.0).vel(0.6).pan(pan));
        if let Some(pip) = s.deg(7 - k, POP_HOME) {
            s.play(Note::new(tick, Atom::Pip, pip).vel(0.4).pan(pan));
        }
        t += pace;
        pace *= 1.18;
        if t > 0.75 * length {
            break;
        }
    }
    let whirr = s.secs(0.7 * length);
    s.play(Note::new(at, Atom::Whirr, 84.0).vel(0.8).hold(whirr));
    if let Some(horn) = s.deg(0, WAH_HOME) {
        let wah = Note::new(at + s.secs(0.6 * length), Atom::Wah, horn);
        let hold = s.secs(0.3 * length);
        s.play(wah.vel(0.8).hold(hold));
    }
}

/// The pickup into a hit at `hit` from `now`: a note at once and, if there
/// is room, one a thirty-second before the hit, rising into the tonic;
/// none when the hit is under [`PICKUP_MIN`] away.
fn pickup(s: &mut Score, now: u64, hit: u64, atom: Atom, floor: i32) {
    if hit < now + s.secs(PICKUP_MIN) {
        return;
    }
    let (Some(first), Some(second)) = (s.deg(-2, floor), s.deg(-1, floor)) else {
        return;
    };
    s.play(Note::new(now, atom, first).vel(0.5).pan(-0.2));
    let late = hit.saturating_sub(s.beats(0.125));
    if late > now + s.beats(0.0625) {
        s.play(Note::new(late, atom, second).vel(0.55).pan(-0.1));
    }
}

/// An arpeggio up the chord from the tonic at `floor`: `n` sixteenths from
/// `at` on `atom`, the last accented; where the last lands.
fn arpeggio(s: &mut Score, at: u64, n: i32, atom: Atom, floor: i32, velocity: f32) -> u64 {
    let mut last = at;
    for i in 1..=n {
        let t = at + s.beats(0.25 * f64::from(i));
        let Some(note) = s.tonal(i, floor) else {
            return at;
        };
        last = t;
        let accent = if i == n { 1.0 } else { 0.8 };
        let pan = -0.35 + 0.7 * i as f32 / n.max(1) as f32;
        s.play(Note::new(t, atom, note).vel(velocity * accent).pan(pan));
    }
    last
}

/// A bell's tremolo: `n` strikes in thirty-seconds from `at`, between the
/// tonic and the fifth two octaves up, fading.
fn tremolo(s: &mut Score, at: u64, n: i32, velocity: f32) {
    let mut v = velocity;
    for i in 0..n {
        let t = at + s.beats(0.125 * f64::from(i));
        let (k, pan) = if i % 2 == 0 { (6, -0.25) } else { (5, 0.25) };
        let note = s.bright(k, STING_HOME);
        s.play(Note::new(t, Atom::Chime, note).vel(v).pan(pan));
        v *= 0.88;
    }
}

/// Glitter: `n` sparkles in thirty-seconds from `at`, high up the
/// pentatonic, anywhere across.
fn glitter(s: &mut Score, at: u64, n: i32, velocity: f32, tag: Tag) {
    for i in 0..n {
        let t = at + s.beats(0.125 * f64::from(i));
        let d = DEGREES + (s.unit() * 2.0 * DEGREES as f32) as i32;
        let pan = s.scatter(0.8);
        let note = s.deg(d, BELL_HOME).unwrap_or(HIGH_NOTES[i as usize % 4]);
        let grain = Note::new(t, Atom::Glitter, note).vel(velocity);
        s.play(grain.pan(pan).tag(tag));
    }
}

/// The chord's first `n` tones over the tonic at `floor`, each `like` but
/// on its own tone and spread across (none with no key).
fn chord(s: &mut Score, like: Note, n: i32, floor: i32) {
    for k in 0..n {
        let Some(pitch) = s.tonal(k, floor) else {
            return;
        };
        let pan = -0.4 + 0.8 * k as f32 / (n - 1).max(1) as f32;
        s.play(Note { pitch, ..like }.pan(pan));
    }
}

/// Brass: both layers of the chord's first `n` tones, each `like`.
fn brass(s: &mut Score, like: Note, n: i32) {
    for atom in BRASS {
        chord(s, Note { atom, ..like }, n, STING_HOME);
    }
}

/// A brass note at `at`, at `velocity`, held `length` (its pitch the
/// chord's).
fn horn(at: u64, velocity: f32, length: u64) -> Note {
    Note::new(at, BRASS[0], 0.0).vel(velocity).hold(length)
}

/// The register's ka-ching at `at`: the lever's clack and thud, the bell
/// struck twice a hair apart so it shimmers, a fifth over it, the drawer
/// sliding out, its coins rattling.
fn ka_ching(s: &mut Score, at: u64) {
    s.play(Note::new(at, Atom::Clack, 60.0).pan(-0.1));
    s.play(
        Note::new(at + s.secs(0.035), Atom::Clack, 64.0)
            .vel(0.7)
            .pan(0.1),
    );
    s.play(Note::new(at, Atom::Thud, 43.0).vel(0.8));
    let ching = at + s.secs(0.07);
    let bell = s.bright(6, STING_HOME);
    s.play(Note::new(ching, Atom::Ching, bell).vel(0.9).pan(-0.15));
    s.play(
        Note::new(ching, Atom::Ching, bell + 0.07)
            .vel(0.7)
            .pan(0.15),
    );
    let fifth = s.bright(8, STING_HOME);
    s.play(Note::new(ching + s.secs(0.004), Atom::Ching, fifth).vel(0.4));
    let slide = s.secs(0.24);
    s.play(
        Note::new(at + s.secs(0.08), Atom::Drawer, 60.0)
            .vel(0.6)
            .hold(slide),
    );
    for k in 0..6 {
        let t = at + s.secs(0.1 + 0.04 * f64::from(k));
        let pan = s.scatter(0.6);
        clink(s, t, k as usize, 0.8 - 0.08 * k as f32, pan);
    }
}

/// A rung's stinger: from `now`, its hit at `hit`, `share` of the rungs
/// up (1: the top), with the rung's `layers`; where it ends.
pub(super) fn stinger(s: &mut Score, now: u64, hit: u64, share: f32, layers: &[Streak]) -> u64 {
    let v = 0.6 + 0.4 * share;
    let octave = layers.contains(&Streak::Octave);
    pickup(s, now, hit, Atom::Pluck, STING_HOME);
    if let Some(tonic) = s.tonal(0, STING_HOME) {
        s.play(Note::new(hit, Atom::Pluck, tonic).vel(v));
        if octave {
            s.play(Note::new(hit, Atom::Pluck, tonic - 12.0).vel(0.7 * v));
        }
    }
    if octave {
        let bell = s.bright(3, STING_HOME);
        s.play(
            Note::new(hit, Atom::Chime, bell)
                .vel(0.4 + 0.4 * share)
                .pan(0.2),
        );
    }
    let n = 1 + (5.0 * share).round() as i32;
    let end = arpeggio(s, hit, n, Atom::Pluck, STING_HOME, v).max(hit + s.beats(0.5));
    if layers.contains(&Streak::Register) {
        let bells = (8.0 * share).round() as i32;
        tremolo(s, hit + s.beats(0.25), bells, 0.35);
        ka_ching(s, hit + s.secs(0.06));
    }
    if layers.contains(&Streak::Brass) {
        let held = s.beats(1.0);
        brass(s, horn(hit, 0.55, held), 3);
    }
    if layers.contains(&Streak::Glitter) {
        glitter(s, hit + s.beats(0.5), 6, 0.5, Tag::NONE);
    }
    end + s.beats(0.5)
}

/// A phrase's chime (or the diploma's points) at `tier` (none: under the
/// first): from `now`, its hit at `hit`, with the tier's layers, bigger
/// the higher it is; where it ends.
pub(super) fn chime(s: &mut Score, now: u64, hit: u64, tier: Option<Tier>) -> u64 {
    let share = tier_share(tier);
    let has = |layer| chime_has(tier, layer);
    let v = 0.5 + 0.4 * share;
    pickup(s, now, hit, Atom::Chime, BELL_HOME);
    let bells = 2 + (2.0 * share).round() as i32;
    for k in 0..bells {
        let t = hit + s.beats(0.0625 * f64::from(k));
        let pan = -0.3 + 0.6 * k as f32 / (bells - 1).max(1) as f32;
        let note = s.bright(k, BELL_HOME);
        s.play(Note::new(t, Atom::Chime, note).vel(v).pan(pan));
    }
    let mut end = hit + s.beats(1.0);
    if has(Chime::Plucks) {
        let n = (5.0 * share).round() as i32;
        end = end.max(arpeggio(s, hit, n, Atom::Pluck, STING_HOME, 0.8 * v));
    }
    // Its brass, as its layers have it: the short hit, or a "ta-daa".
    for b in chime_brass(tier) {
        let at = hit.saturating_sub(s.beats(b.before));
        if at < now {
            continue;
        }
        let held = s.beats(b.length);
        brass(s, horn(at, b.velocity, held), b.tones);
        end = end.max(at + held);
    }
    if has(Chime::Choir) {
        let long = s.beats(2.0);
        let choir = Note::new(hit, Atom::Choir, 0.0).vel(0.7).hold(long);
        chord(s, choir, 4, STING_HOME);
        if let Some(low) = s.tonal(0, LOW_HOME) {
            s.play(Note::new(hit, Atom::Low, low).vel(0.7));
        }
        glitter(s, hit + s.beats(0.25), 8, 0.45, Tag::NONE);
        end = end.max(hit + long);
    }
    end
}

/// Coins pouring from `at`: `per_beat` a beat for `beats` beats, on the
/// grid with a little jitter, anywhere across, thinning to the end.
fn coins(s: &mut Score, at: u64, per_beat: u32, beats: u32, velocity: f32, tag: Tag) {
    let n = per_beat * beats;
    for i in 0..n {
        let t = at + s.beats(f64::from(i) / f64::from(per_beat));
        let thin = 1.0 - 0.5 * i as f32 / n.max(1) as f32;
        let k = (s.unit() * 6.0) as usize;
        let pan = s.scatter(0.7);
        let v = velocity * thin * (0.75 + 0.25 * s.unit());
        let atom = COINS[k % COINS.len()];
        let d = CLINK_DEGREES[k % CLINK_DEGREES.len()];
        let note = s.deg(d, CLINK_HOME).unwrap_or(HIGH_NOTES[k % 2]);
        s.play(Note::new(t, atom, note).vel(v).pan(pan).tag(tag));
    }
}

/// The top rung's payout on the downbeat `at` (`now` the event's frame),
/// bars of `bar` beats, all of it within [`payout_beats`]; led into by a
/// whoosh [`PAYOUT_LEAD`] before it where `lead` (that beat clear of the
/// singing).
pub(super) fn payout(s: &mut Score, now: u64, at: u64, bar: u32, lead: bool) {
    let bar = bar.max(2);
    let lead_len = s.beats(PAYOUT_LEAD);
    if lead && at >= now + lead_len {
        let whoosh = Note::new(at - lead_len, Atom::Whoosh, 72.0).vel(0.6);
        s.play(whoosh.hold(lead_len).tag(PAYOUT));
    }
    // Bells climbing in eighths, quickening into sixteenths a beat before
    // the win's bar, a beat's breath before the win.
    let eighths = 2 * (bar - 2).max(1);
    let climb = (0..eighths)
        .map(|i| 0.5 * f64::from(i))
        .chain((0..4).map(|i| 0.5 * f64::from(eighths) + 0.25 * f64::from(i)));
    for (k, beats) in climb.enumerate() {
        let k = k as i32;
        let t = at + s.beats(beats);
        let note = s.bright(k, STING_HOME);
        let bell = Note::new(t, Atom::Chime, note).vel(0.45 + 0.03 * k as f32);
        s.play(bell.pan(-0.5 + 0.08 * k as f32).tag(PAYOUT));
    }
    coins(s, at, 4, bar, 0.7, PAYOUT);
    let win = at + s.beats(f64::from(bar));
    let held = s.beats(1.5);
    brass(s, horn(win, 0.7, held).tag(PAYOUT), 4);
    for k in 3..6 {
        let pan = 0.4 - 0.2 * (k - 3) as f32;
        let note = s.bright(k, BELL_HOME);
        s.play(
            Note::new(win, Atom::Chime, note)
                .vel(0.4)
                .pan(pan)
                .tag(PAYOUT),
        );
    }
    if let Some(low) = s.tonal(0, LOW_HOME) {
        s.play(Note::new(win, Atom::Low, low).vel(0.8).tag(PAYOUT));
    }
    glitter(s, win + s.beats(0.125), 8, 0.45, PAYOUT);
    coins(s, win, 2, bar, 0.45, PAYOUT);
}

/// The top tier's jackpot on the downbeat `at`, within [`JACKPOT_BEATS`].
pub(super) fn jackpot(s: &mut Score, at: u64) {
    for atom in SLAM {
        s.play(Note::new(at, atom, 60.0));
    }
    let long = s.beats(JACKPOT_BEATS - 0.5);
    let choir = Note::new(at, Atom::Choir, 0.0).vel(0.75).hold(long);
    chord(s, choir, 4, STING_HOME);
    let held = s.beats(1.0);
    brass(s, horn(at, 0.6, held), 3);
    if let Some(low) = s.tonal(0, LOW_HOME) {
        s.play(Note::new(at, Atom::Low, low).vel(0.8));
    }
    tremolo(s, at + s.beats(0.25), 8, 0.4);
    coins(s, at + s.beats(0.5), 4, 2, 0.6, Tag::NONE);
    for i in 0..8 {
        let t = at + s.beats(1.0 + 0.125 * f64::from(i));
        let pan = s.scatter(0.7);
        let note = s
            .deg(DEGREES + i, BELL_HOME)
            .unwrap_or(HIGH_NOTES[i as usize % 4]);
        s.play(Note::new(t, Atom::Glitter, note).vel(0.5).pan(pan));
    }
}

/// The riser from `at` into a high rung: air rising and swelling, held at
/// most [`RISE_WAIT`] beats (if no rung comes it ends by itself).
pub(super) fn riser(s: &mut Score, at: u64) {
    let hold = s.beats(RISE_WAIT);
    s.play(Note::new(at, Atom::Riser, 84.0).hold(hold).tag(RISER));
}

/// The riser let go a breath before the hit at `hit` it led into.
pub(super) fn riser_lands(s: &mut Score, hit: u64) {
    let breath = s.secs(BREATH);
    s.release(hit.saturating_sub(breath), RISER);
}

/// The held sounds let go at `at`: the riser and the shimmer.
pub(super) fn let_go(s: &mut Score, at: u64) {
    s.release(at, RISER);
    s.release(at, SHIMMER);
}

/// The shimmer pad from `at` (the tonic and the fifth high up; none with
/// no key), held [`SHIMMER_BARS`] from there: started, or held on (its
/// voices take the note again where they are, so it carries on).
pub(super) fn shimmer(s: &mut Score, at: u64) {
    let hold = s.beats(SHIMMER_BARS * f64::from(s.bar));
    for k in 0..2u8 {
        let Some(note) = s.tonal(2 * i32::from(k), PAD_HOME) else {
            return;
        };
        let pan = if k == 0 { -0.3 } else { 0.3 };
        let pad = Note::new(at, Atom::Pad, note).line(k).tag(SHIMMER);
        s.play(pad.vel(0.8).pan(pan).hold(hold));
    }
}

/// Glitter over a word at the top: a grain on each of `steps` (the next
/// two sixteenths), every other one on average.
pub(super) fn glitter_after(s: &mut Score, steps: [u64; 2]) {
    for (i, t) in steps.into_iter().enumerate() {
        if s.unit() < 0.5 {
            let d = 2 * DEGREES + (s.unit() * 5.0) as i32;
            let pan = 1.6 * s.unit() - 0.8;
            let note = s.deg(d, PAD_HOME - 12).unwrap_or(HIGH_NOTES[2 + i % 2]);
            let grain = Note::new(t, Atom::Glitter, note).vel(0.3);
            s.play(grain.pan(pan).tag(STREAK));
        }
    }
}

/// Tick `k` of `of` of the diploma's count-up at `at`: a tick, and a pip
/// rising up the pentatonic over the count (two octaves).
pub(super) fn tick(s: &mut Score, at: u64, k: u32, of: u32) {
    let d = (2 * DEGREES) as u32 * k / of.max(1);
    s.play(Note::new(at, Atom::Tick, 60.0).vel(0.6));
    if let Some(pip) = s.deg(d as i32, POP_HOME) {
        s.play(Note::new(at, Atom::Pip, pip).vel(0.4));
    }
}

/// A step of a song's preparation found something at `at`, the `step`th
/// (0 to [`crate::core::prep::LAST_STEP`]): a soft pluck up the pentatonic
/// for each later step, a glint an octave over it a moment after.
pub(super) fn found(s: &mut Score, at: u64, step: u32) {
    let k = 2 + step.min(crate::core::prep::LAST_STEP) as i32;
    let note = s.bright(k, BELL_HOME - 12);
    s.play(Note::new(at, Atom::Pluck, note).vel(0.45));
    let glint = Note::new(at + s.secs(0.05), Atom::Glint, note + 12.0);
    s.play(glint.vel(0.25));
}

/// A diploma's gem popping in at `at`, `share` of the tiers up, at
/// `pan`: a bell higher for a higher tier, a sparkle over it.
pub(super) fn gem(s: &mut Score, at: u64, share: f32, pan: f32) {
    let k = 3 + (3.0 * share).round() as i32;
    let bell = s.bright(k, BELL_HOME - 12);
    s.play(Note::new(at, Atom::Chime, bell).vel(0.6).pan(pan));
    let sparkle = s.bright(k + 3, BELL_HOME - 12);
    s.play(Note::new(at, Atom::Glitter, sparkle).vel(0.4).pan(pan));
}
