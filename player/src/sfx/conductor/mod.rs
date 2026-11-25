//! The sound conductor: the karaoke game's sounds, planned live in the
//! song's tempo and key, as timed notes for our own synthesizer
//! ([`jodel_synth::Synth`], made with the conductor: [`Conductor::new`]).
//!
//! The conductor plans on the thread that reads the game (where the
//! moments are, [`super::moments`]), in the song's stream frames, the
//! frames a sound effect is cued on. It is told the game's events with
//! their frames ([`Conductor::event`], which says where its hit lands, so
//! a look can land with it) and the music clock ([`Conductor::set_clock`]:
//! the song's beat grid from the analysis and the sung lines, [`Clock`]);
//! it sends the notes, the releases of the held ones and the calls to
//! call some off ([`jodel_synth::Command`]) to wherever they go
//! ([`jodel_synth::Commands`]): the synth itself when rendering offline,
//! else a queue to it on the effects path, which puts the stream's frames
//! on the synth's as it hands them over.
//!
//! - At once: a word's pop on the pitch ladder (the song's pentatonic, a
//!   step higher for each word in a row, the tonic on each rung), its
//!   clink from the high rungs on (round-robin, a little different each
//!   time, softer when they come thick, none while a stinger rings), the
//!   rescue shot's zap, the wind-down of a streak lost.
//! - On the next beat (a pickup at once): a rung's stinger, a phrase's
//!   chime, the diploma's points.
//! - On a downbeat with room before the next sung line: the top rung's
//!   payout, the top tier's jackpot.
//! - Held: a riser into a high rung from the word before it, a shimmer
//!   pad while the streak is high, glitter after each word at the top; a
//!   streak lost lets them go (and calls off a payout still to come).
//!
//! Where a hit lands is the song's grid's one rule
//! ([`jodel_rhythm::grid::Grid::land`]), the one the looks land by too.
//! The rungs are reached by the words: a [`Event::Word`] whose run reaches
//! a rung higher than the last reached plays that rung's stinger after its
//! pop, so a word and its rung are one event and a jump in the run misses
//! none. Tiers and rungs are plain indices into a [`Layout`] (how many
//! tiers, at how many words in a row each rung is): what a sound is follows
//! from its share of the way to the top.
//!
//! The key is the song's as the analysis found it, shifted as it is heard;
//! with none, the sounds stay unpitched or high over the song. Telling it
//! events and the clock allocates nothing.

mod clock;
mod key;
mod ladder;
mod palette;
mod sounds;

#[cfg(test)]
mod tests;

use jodel_rhythm::grid::When;
use jodel_synth::chorus::ChorusSettings;
use jodel_synth::delay::DelaySettings;
use jodel_synth::hall::HallSettings;
use jodel_synth::{Command, Commands, Synth};

pub use clock::Clock;
pub(crate) use sounds::JACKPOT_BEATS;

use key::Pentatonic;
use palette::Palette;
use sounds::{Score, HIGH, HIGHER, PAYOUT, PAYOUT_LEAD, STREAK};

/// The synth's voices, and the notes that can wait in it.
const VOICES: usize = 96;
const CUES: usize = 1_024;
/// The most ticks a count-up plays.
const MAX_TICKS: u32 = 64;
/// The most rungs a layout has.
const MAX_RUNGS: usize = 16;
/// Clinks closer than this, seconds, come softer.
const CLINK_GAP: f64 = 0.2;

/// The most commands one event (or a clock or a cancel) sends: what a
/// queue to the synth must hold at least for one at a time (the longest
/// count-up, two notes a tick, is the most).
pub const MOST_COMMANDS: usize = 160;

/// How many tiers there are and at how many words in a row each rung is
/// reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    tiers: usize,
    rungs: [u32; MAX_RUNGS],
    count: usize,
}

impl Layout {
    /// `tiers` tiers, rungs at `rungs` words in a row.
    ///
    /// # Panics
    ///
    /// With no tier, no rung or more than 16, or rungs not strictly
    /// increasing from 1 up.
    pub fn new(tiers: usize, rungs: &[u32]) -> Layout {
        assert!(tiers > 0, "a layout needs a tier");
        assert!(
            !rungs.is_empty() && rungs.len() <= MAX_RUNGS,
            "1 to {MAX_RUNGS} rungs"
        );
        assert!(rungs[0] >= 1, "a rung needs a word");
        assert!(rungs.windows(2).all(|w| w[0] < w[1]), "rungs increasing");
        let mut r = [0; MAX_RUNGS];
        r[..rungs.len()].copy_from_slice(rungs);
        Layout {
            tiers,
            rungs: r,
            count: rungs.len(),
        }
    }

    /// The rungs' words in a row.
    fn rungs(&self) -> &[u32] {
        &self.rungs[..self.count]
    }

    /// Rung `r`'s share of the way to the top (1 up to the rungs).
    fn rung_share(&self, r: u32) -> f32 {
        (r as f32 / self.count as f32).clamp(0.0, 1.0)
    }

    /// Tier `t`'s (0 up).
    fn tier_share(&self, t: usize) -> f32 {
        ((t + 1) as f32 / self.tiers as f32).clamp(0.0, 1.0)
    }
}

/// A moment of the diploma.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Diploma {
    /// The points counting up in `ticks` ticks, in tempo (at most 64).
    Count {
        /// Its ticks.
        ticks: u32,
    },
    /// A gem popping in, of a tier.
    Gem(usize),
    /// The points all counted: the tier's chime (none: the smallest).
    Points(Option<usize>),
}

/// A game event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    /// A word's bubble popped: the words in a row with it (0: none, no
    /// pop; reaching a rung higher than the last: its stinger too) and
    /// where across (-1 to 1).
    Word {
        /// Words in a row.
        streak: u32,
        /// Its pan.
        pan: f32,
    },
    /// The rescue shot fired, from where across.
    Zap {
        /// Its pan.
        pan: f32,
    },
    /// The streak lost (from the rung last reached).
    Lost,
    /// A phrase rated, its tier (0 the first).
    Phrase(usize),
    /// The top tier's jackpot.
    Jackpot,
    /// A moment of the diploma.
    Diploma(Diploma),
}

/// The sound conductor: the planner.
pub struct Conductor {
    palette: Palette,
    layout: Layout,
    clock: Clock,
    rate: u32,
    key: Option<Pentatonic>,
    /// The tempo last told the synth.
    tempo: f64,
    seed: u32,
    /// The commands not taken.
    dropped: u64,
    /// The clinks played, the last one's frame, and the frame until which
    /// a stinger rings (no clinks).
    clinks: usize,
    last_clink: Option<u64>,
    quiet_until: u64,
    /// The rung the run has reached (0: none), and whether the riser into
    /// the next one sounds.
    rung: u32,
    riser: bool,
}

impl Conductor {
    /// A conductor planning at `rate` Hz for `layout`'s tiers and rungs (at
    /// the free tempo, no key known), and the synth its notes are for, its
    /// bus set (to be moved to where it renders: a synth on the effects
    /// path, at the device's rate).
    ///
    /// # Panics
    ///
    /// If the synth takes no such rate (32 to 192 kHz).
    pub fn new(rate: u32, layout: Layout) -> (Conductor, Synth) {
        let (bank, palette) = Palette::bank();
        let mut synth = Synth::with_cues(rate, bank, VOICES, CUES);
        synth.set_reverb(HallSettings {
            size: 0.5,
            decay: 1.8,
            wet: 0.3,
            ..HallSettings::default()
        });
        synth.set_delay(DelaySettings {
            wet: 0.35,
            ..DelaySettings::default()
        });
        synth.set_chorus(ChorusSettings::default());
        let clock = Clock::free(rate);
        synth.set_tempo(clock.tempo() as f32);
        let conductor = Conductor {
            palette,
            layout,
            tempo: clock.tempo(),
            clock,
            rate,
            key: None,
            seed: 0x9E37_79B9,
            dropped: 0,
            clinks: 0,
            last_clink: None,
            quiet_until: 0,
            rung: 0,
            riser: false,
        };
        (conductor, synth)
    }

    /// The music clock from now on; the synth told its tempo when it
    /// changed.
    pub fn set_clock(&mut self, out: &mut impl Commands, clock: Clock) {
        self.clock = clock;
        if (clock.tempo() - self.tempo).abs() > 0.5 {
            self.tempo = clock.tempo();
            let bpm = self.tempo as f32;
            self.score(out).send(Command::Tempo { bpm });
        }
    }

    /// The song's key as the analysis found it (none: unpitched and high),
    /// heard `shift` semitones up (down when negative).
    pub fn set_key(&mut self, key: Option<&jodel_rhythm::key::Key>, shift: i8) {
        self.key = key.map(|k| Pentatonic::of(k, shift));
    }

    /// The commands sent that were not taken (a full queue): what was
    /// lost; a held sound let go by one ends on its own soon after.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Everything called off from `at` on (a seek, the song left): what
    /// waits is dropped, what is held let go; what sounds rings out.
    pub fn cancel(&mut self, out: &mut impl Commands, at: u64) {
        let mut s = self.score(out);
        s.send(Command::Cancel { tag: None });
        sounds::let_go(&mut s, at);
        self.rung = 0;
        self.riser = false;
        self.quiet_until = 0;
        self.last_clink = None;
    }

    /// `event` on frame `at`, its notes sent to `out`; the frame its main
    /// hit lands on.
    pub fn event(&mut self, out: &mut impl Commands, at: u64, event: Event) -> u64 {
        match event {
            Event::Word { streak, pan } => self.word(out, at, streak, pan),
            Event::Zap { pan } => {
                sounds::zap(&mut self.score(out), at, pan);
                at
            }
            Event::Lost => {
                let share = self.layout.rung_share(self.rung);
                self.lose(out, at);
                sounds::wind_down(&mut self.score(out), at, share);
                at
            }
            Event::Phrase(tier) => self.chime(out, at, self.layout.tier_share(tier)),
            Event::Jackpot => {
                let length = JACKPOT_BEATS * self.clock.beat_len();
                let d = self.clock.land(at, When::Downbeat { length });
                sounds::jackpot(&mut self.score(out), d);
                d
            }
            Event::Diploma(Diploma::Count { ticks }) => {
                let ticks = ticks.min(MAX_TICKS);
                let start = self.clock.land(at, When::Beat);
                let clock = self.clock;
                let mut s = self.score(out);
                for k in 0..ticks {
                    sounds::tick(&mut s, clock.tick(start, k, ticks), k, ticks);
                }
                clock.tick(start, ticks.saturating_sub(1), ticks)
            }
            Event::Diploma(Diploma::Gem(tier)) => {
                let share = self.layout.tier_share(tier);
                sounds::gem(&mut self.score(out), at, share);
                at
            }
            Event::Diploma(Diploma::Points(tier)) => {
                let share = tier.map_or(0.0, |t| self.layout.tier_share(t));
                self.chime(out, at, share)
            }
        }
    }

    /// A chime `share` of the tiers up from `at`, on the next beat; where
    /// its hit lands.
    fn chime(&mut self, out: &mut impl Commands, at: u64, share: f32) -> u64 {
        let hit = self.clock.land(at, When::Beat);
        sounds::chime(&mut self.score(out), at, hit, share);
        hit
    }

    /// The streak gone at `at`: the payout to come and the glitter called
    /// off, what is held let go.
    fn lose(&mut self, out: &mut impl Commands, at: u64) {
        let mut s = self.score(out);
        s.send(Command::Cancel { tag: Some(PAYOUT) });
        s.send(Command::Cancel { tag: Some(STREAK) });
        sounds::let_go(&mut s, at);
        self.rung = 0;
        self.riser = false;
    }

    /// A word's bubble popped on `at`, `streak` in a row with it; the
    /// frame its pop (or the rung it reached) lands on.
    fn word(&mut self, out: &mut impl Commands, at: u64, streak: u32, pan: f32) -> u64 {
        if streak == 0 {
            return at;
        }
        let step = ladder::step(streak, self.layout.rungs());
        if step.rung < self.rung {
            // A new run the game did not say the last was lost before.
            self.lose(out, at);
        }
        let share = self.layout.rung_share(step.rung);
        sounds::pop(&mut self.score(out), at, step, share, pan);
        let second = f64::from(self.rate);
        if share >= HIGH && at >= self.quiet_until {
            let thick = self
                .last_clink
                .is_some_and(|t| at.saturating_sub(t) < (CLINK_GAP * second) as u64);
            let velocity = if thick { 0.45 } else { 0.9 };
            let k = self.clinks;
            sounds::clink(&mut self.score(out), at, k, velocity, pan);
            self.clinks += 1;
            self.last_clink = Some(at);
        }
        let mut lands = at;
        if step.rung > self.rung {
            lands = self.rung_up(out, at, step.rung);
        } else {
            // The word before a high rung: the riser, until the rung comes.
            let rungs = self.layout.rungs();
            let next = rungs.iter().position(|&r| r == streak + 1);
            let high = next.is_some_and(|k| self.layout.rung_share(k as u32 + 1) >= HIGHER);
            if high && !self.riser {
                sounds::riser(&mut self.score(out), at);
                self.riser = true;
            }
            if step.rung as usize == self.layout.count {
                let first = self.clock.next_step(at + 1, 4);
                let sixteenth = (self.clock.beat_len() / 4.0) as u64;
                let steps = [first, first + sixteenth];
                sounds::glitter_after(&mut self.score(out), steps);
            }
        }
        if share >= HIGHER {
            // Started on the rung's hit; each word after holds it on.
            sounds::shimmer(&mut self.score(out), lands);
        }
        lands
    }

    /// Rung `r` reached on `at`, above the last: its stinger, and at the
    /// top the payout; the frame its hit lands on.
    fn rung_up(&mut self, out: &mut impl Commands, at: u64, r: u32) -> u64 {
        self.rung = r;
        let share = self.layout.rung_share(r);
        let hit = self.clock.land(at, When::Beat);
        if self.riser {
            sounds::riser_lands(&mut self.score(out), hit);
            self.riser = false;
        }
        let end = sounds::stinger(&mut self.score(out), at, hit, share);
        self.quiet_until = self.quiet_until.max(end);
        if r as usize >= self.layout.count {
            let bar = self.clock.bar();
            let beat = self.clock.beat_len();
            let length = sounds::payout_beats(bar) * beat;
            let d = self.clock.land(end, When::Downbeat { length });
            let lead = (PAYOUT_LEAD * beat) as u64;
            let room = d >= end + lead && self.clock.clear(d - lead, lead);
            sounds::payout(&mut self.score(out), end, d, bar, room);
        }
        hit
    }

    /// The score the sounds are written into, sending to `out`.
    fn score<'a>(&'a mut self, out: &'a mut dyn Commands) -> Score<'a> {
        Score {
            out,
            dropped: &mut self.dropped,
            p: &self.palette,
            key: self.key,
            beat: self.clock.beat_len(),
            bar: self.clock.bar(),
            rate: f64::from(self.rate),
            seed: &mut self.seed,
        }
    }
}
