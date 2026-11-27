//! The sound conductor (an adaptive sound-effect scheduler): the karaoke
//! game's sounds, planned live in the
//! song's tempo and key, as timed notes for our own synthesizer
//! ([`jodel_synth::Synth`], made with the conductor: [`Conductor::new`];
//! its bank every theme's atoms, [`bank`], so a theme is the patches the
//! conductor plays: [`Conductor::set_theme`]).
//!
//! The conductor plans on the thread that reads the game (where the
//! moments are, [`super::moments`]), in the frames its sounds are cued on:
//! the song's stream frames while it is sung, the desk's own after it (the
//! diploma). It is told the game's events with their frames
//! ([`Conductor::event`], which says where its hit lands, so a look can
//! land with it) and the music clock ([`Conductor::time_by`]: the song's
//! beat grid from the analysis and the sung lines, [`Clock`]; the synth
//! told its tempo by whoever plays on it, [`Tempo`]); it sends the
//! notes, the releases of the held ones and the calls to call some off
//! ([`jodel_synth::Command`]) to wherever they go
//! ([`jodel_synth::Commands`]): the synth itself when rendering offline,
//! else a queue to it on the effects path
//! ([`jodel_output::mixer::Effects::commands`]), which puts those frames on
//! the synth's as it hands them over.
//!
//! - At once: a word's pop on the pitch ladder (the song's pentatonic, a
//!   step higher for each word in a row, the tonic on each rung; louder,
//!   with a body an octave under it, for a word held full long), its
//!   clink at the rungs that have them (round-robin, a little different
//!   each time, softer when they come thick, none while a stinger rings),
//!   the rescue shot's zap, the wind-down of a streak lost.
//! - On the next beat (a pickup at once): a rung's stinger, a phrase's
//!   chime, the diploma's points and its best streak's rung.
//! - On a downbeat with room before the next sung line: the top rung's
//!   payout, the top tier's jackpot.
//! - Where the slam lands, at the rows whose layers call
//!   ([`Streak::Call`], [`Chime::Call`]): its word sung by the announcer
//!   ([`call`]), a rung's multiplier from its stinger's hit, a phrase's
//!   tier's name from the jackpot's downbeat; softer over a sung line.
//! - Held: a riser into a rung from the word before it, a shimmer pad
//!   while the streak is at a rung that holds one, glitter after each word
//!   at one that has it; a streak lost lets them go (and calls off a payout
//!   still to come).
//! - The diploma's as it draws them: each tick of its count-up (a pip a
//!   step higher as the count goes), its gems, and its sheet's own, its
//!   paper, pen, stamps and tunes, at once ([`Sheet`]: each on fixed notes,
//!   the same every time).
//! - The Synth tab's ([`crate::sfx::bench`]): any of the sounds above on
//!   a click, by a conductor of its own on the same synth, and the keys of
//!   its keyboard, each note on its own tag ([`tags`]).
//!
//! Where a hit lands is the song's grid's one rule
//! ([`jodel_rhythm::grid::Grid::land`]), the one the looks land by too.
//! The rungs are reached by the words: a [`Event::Word`] whose run reaches
//! a rung higher than the last reached plays that rung's stinger after its
//! pop, so a word and its rung are one event and a jump in the run misses
//! none. The tiers and their rungs are the tier table's rows
//! ([`crate::tiers::TIERS`]: how many, at how many words in a row each
//! rung is reached): what a sound is made of is its row's layers
//! ([`Layers`]), how big it is its share of the way to the top.
//!
//! The key is the song's as the analysis found it, shifted as it is heard;
//! with none, the sounds stay unpitched or high over the song. The bus's
//! level follows the song's loudness ([`Conductor::set_loudness`],
//! [`calibration`]). Telling it events, the clock and the loudness
//! allocates nothing.

pub mod calibration;
pub mod call;
mod clock;
mod key;
mod ladder;
mod layers;
mod palette;
mod pen;
mod sheet;
mod sounds;
pub(crate) mod tags;

#[cfg(test)]
mod tests;

use jodel_rhythm::grid::When;
use jodel_synth::chorus::ChorusSettings;
use jodel_synth::delay::DelaySettings;
use jodel_synth::dynamics::CompressorSettings;
use jodel_synth::hall::HallSettings;
use jodel_synth::{Bank, Command, Commands, Synth, GENTLE};

pub use clock::Clock;
pub use layers::{at_rung, Chime, Layers, Streak, StreakSet};
pub use palette::{bank, every, id, of, patch, place, Atom};
pub use sheet::{Entry, Sheet};
pub(crate) use sounds::{Note, Score};

use crate::core::singing::rung;
use crate::theme::Theme;
use crate::tiers::{self, Landing, Tier};
use key::Pentatonic;
use layers::{chime_has, share};
use sounds::{Origin, PAYOUT_LEAD};
use tags::{PAYOUT, STREAK};

/// The synth's voices, and the notes that can wait in it.
const VOICES: usize = 96;
const CUES: usize = 1_024;
/// Clinks closer than this, seconds, come softer.
const CLINK_GAP: f64 = 0.2;
/// How far the synth's bus lifts the sounds over a song at the reference
/// loudness ([`calibration::REFERENCE`], a pop track's instrumental), dB: after its gentle compressor (which hears them as the
/// patches set them, so it takes the same edge off), into its limiter at
/// -1 dBFS; a louder song lifts them more, a quieter one less
/// ([`calibration::trim`]). So the game sits on top of the song as a
/// game's sounds do over its music, the Effects strip at 0 dB: a word's
/// pop peaks about -11.5 dBFS, the top phrase's chime and the jackpot
/// about -6; the limiter takes only the biggest pile-ups (the top rung's
/// stinger, its plucks struck together), and more of a loud song's.
pub const LEVEL: f32 = 3.33;

/// The most commands one event (or a clock or a cancel) sends: what a
/// queue to the synth must hold at least for one at a time (the top
/// rung's stinger with its payout is the most).
pub const MOST_COMMANDS: usize = 160;

/// A moment of the diploma.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Diploma {
    /// Tick `k` (from 0) of the points counting up in `of`.
    Tick {
        /// Its place.
        k: u32,
        /// The ticks in all.
        of: u32,
    },
    /// A gem popping in, of a tier (none: a plain one), where across the
    /// sheet (-1 to 1).
    Gem {
        /// Its tier.
        tier: Option<Tier>,
        /// Its pan.
        pan: f32,
    },
    /// The points all counted: the tier's chime (none: the smallest).
    Points(Option<Tier>),
    /// The best streak written in: the stinger of the rung it reached.
    Rung(Tier),
    /// One of the sheet's own sounds, at `gain` (1: at its level) and
    /// `pan` (-1 to 1, where on the sheet it lands).
    Sheet {
        /// Which.
        sound: Sheet,
        /// Its gain.
        gain: f32,
        /// Its pan.
        pan: f32,
    },
}

/// What the announcer calls out ([`Conductor::announce`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Called {
    /// A tier's name ("di-va!").
    Name(Tier),
    /// A tier's rung's multiplier ("times thir-ty-two!").
    Times(Tier),
}

impl Called {
    /// Its word: its tier's name, its row's
    /// ([`crate::tiers::TierSpec::call`]); its multiplier, the number's
    /// ([`call::times`]; none for a number with no word).
    fn word(self) -> Option<call::Word> {
        match self {
            Called::Name(tier) => Some(tier.spec().call),
            Called::Times(tier) => call::times(tier.multiplier()),
        }
    }
}

/// A game event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    /// A word's bubble popped: the words in a row with it (0: none, no
    /// pop; reaching a rung higher than the last: its stinger too), where
    /// across (-1 to 1) and how big (0 as any to 1: held full the longest,
    /// [`crate::core::singing::Burst::size`]).
    Word {
        /// Words in a row.
        streak: u32,
        /// Its pan.
        pan: f32,
        /// How big.
        size: f32,
    },
    /// The rescue shot fired, from where across.
    Zap {
        /// Its pan.
        pan: f32,
    },
    /// The streak lost from the rung of a tier.
    Lost(Tier),
    /// A phrase rated, its tier.
    Phrase(Tier),
    /// The top tier's jackpot.
    Jackpot,
    /// A moment of the diploma.
    Diploma(Diploma),
    /// A step of a song's preparation found something: its place among
    /// the steps (the beat 0 … the story's scenes,
    /// [`crate::core::prep::Found::step`]).
    Found {
        /// The step.
        step: u32,
    },
}

/// The sound conductor: the planner.
pub struct Conductor {
    /// The theme whose patches it plays (its place: [`place`]).
    theme: usize,
    clock: Clock,
    rate: u32,
    key: Option<Pentatonic>,

    seed: u32,
    /// The commands not taken.
    dropped: u64,
    /// The clinks played, the last one's frame, and the frame until which
    /// a stinger rings (no clinks).
    clinks: usize,
    last_clink: Option<u64>,
    quiet_until: u64,
    /// The rung the run has reached (none: under the first), and whether
    /// the riser into the next one sounds.
    rung: Option<Tier>,
    riser: bool,
    /// The bus's level told the synth, dB from [`LEVEL`]
    /// ([`calibration::trim`]).
    trim: f32,
}

impl Conductor {
    /// A conductor planning at `rate` Hz for the tiers' rows in `theme`
    /// (at the free tempo, no key known, the song as at the reference
    /// loudness), and the synth its notes are for, every theme's patches
    /// as built ([`bank`], [`Theme::patch`]), made as [`Conductor::synth`]
    /// makes it.
    ///
    /// # Panics
    ///
    /// If the synth takes no such rate (32 to 192 kHz).
    pub fn new(rate: u32, theme: &Theme) -> (Conductor, Synth) {
        let synth = Conductor::synth(rate, bank(|t, a| t.patch(a)));
        (Conductor::planner(rate, theme), synth)
    }

    /// The synth a conductor's notes are for, at `rate` Hz, playing `bank`
    /// (every theme's atoms: [`bank`]), its bus set (lifted by [`LEVEL`]:
    /// the effects path adds no gain of its own; to be moved to where it
    /// renders: a synth on the effects path, at the device's rate), at the
    /// free clock's tempo.
    ///
    /// # Panics
    ///
    /// If the synth takes no such rate (32 to 192 kHz).
    pub fn synth(rate: u32, bank: Bank) -> Synth {
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
        synth.set_compressor(CompressorSettings {
            makeup: LEVEL,
            ..GENTLE
        });
        synth.set_tempo(Tempo::default().told as f32);
        synth
    }

    /// The conductor alone, as [`Conductor::new`] makes it (what it plans
    /// can be written down without a synth: which atoms a sound plays; or
    /// sent to a synth made apart, [`Conductor::synth`]).
    pub fn planner(rate: u32, theme: &Theme) -> Conductor {
        let clock = Clock::free(rate);
        Conductor {
            theme: place(theme),
            clock,
            rate,
            key: None,
            seed: 0x9E37_79B9,
            dropped: 0,
            clinks: 0,
            last_clink: None,
            quiet_until: 0,
            rung: None,
            riser: false,
            trim: 0.0,
        }
    }

    /// The theme whose patches it plays from now on (the synth has every
    /// theme's: what sounds of another rings on).
    pub fn set_theme(&mut self, theme: &Theme) {
        self.theme = place(theme);
    }

    /// The rate it plans at, Hz.
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// The music clock its notes are timed by from now on (the synth is
    /// told its tempo by whoever plays on it: [`Tempo::time`]).
    pub fn time_by(&mut self, clock: Clock) {
        self.clock = clock;
    }

    /// The song's key as the analysis found it (none: unpitched and high),
    /// heard `shift` semitones up (down when negative).
    pub fn set_key(&mut self, key: Option<&jodel_rhythm::key::Key>, shift: i8) {
        self.key = key.map(|k| Pentatonic::of(k, shift));
    }

    /// The song's loudness as played, LUFS (none: not measured yet, or
    /// silent: as at the reference): the bus's level told the synth when it
    /// changes ([`calibration::trim`]; the synth glides to it, before its
    /// limiter). One not taken (a full queue) is told again on the next
    /// call.
    pub fn set_loudness(&mut self, out: &mut impl Commands, lufs: Option<f64>) {
        let trim = calibration::trim(lufs);
        if trim != self.trim && self.score(out).send(Command::Level { db: trim }) {
            self.trim = trim;
        }
    }

    /// The bus's level told the synth, dB from [`LEVEL`].
    pub fn trim(&self) -> f32 {
        self.trim
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
        self.rung = None;
        self.riser = false;
        self.quiet_until = 0;
        self.last_clink = None;
    }

    /// `event` on frame `at`, its notes sent to `out`; the frame its main
    /// hit lands on.
    pub fn event(&mut self, out: &mut impl Commands, at: u64, event: Event) -> u64 {
        match event {
            Event::Word { streak, pan, size } => self.word(out, at, streak, (pan, size)),
            Event::Zap { pan } => {
                sounds::zap(&mut self.score(out), at, pan);
                at
            }
            Event::Lost(from) => {
                self.lose(out, at);
                let reels = layers::reach(from, Streak::Reels);
                sounds::wind_down(&mut self.score(out), at, reels);
                at
            }
            Event::Phrase(tier) => {
                let hit = self.chime(out, at, Some(tier));
                if chime_has(Some(tier), Chime::Call) {
                    let d = self.lands(at, Landing::Phrase);
                    self.announce(out, at, d, Called::Name(tier));
                }
                hit
            }
            Event::Jackpot => {
                let d = self.lands(at, Landing::Phrase);
                sounds::jackpot(&mut self.score(out), d);
                d
            }
            Event::Diploma(Diploma::Tick { k, of }) => {
                sounds::tick(&mut self.score(out), at, k, of);
                at
            }
            Event::Diploma(Diploma::Gem { tier, pan }) => {
                sounds::gem(&mut self.score(out), at, share(tier), pan);
                at
            }
            Event::Diploma(Diploma::Points(tier)) => self.chime(out, at, tier),
            Event::Diploma(Diploma::Rung(tier)) => {
                let hit = self.clock.land(at, When::Beat);
                let (played, size) = (at_rung(Some(tier)), share(Some(tier)));
                sounds::stinger(&mut self.score(out), at, hit, size, played);
                hit
            }
            Event::Diploma(Diploma::Sheet { sound, gain, pan }) => {
                // Its notes' seeds its own: the same every time; its notes
                // from `at`, at its level and gain, panned by `pan`.
                let mut seed = sound.seed();
                let gain = gain * 10f32.powf(sound.level().min(0.0) / 20.0);
                let mut s = Score {
                    seed: &mut seed,
                    origin: Origin { at, gain, pan },
                    ..self.score(out)
                };
                sheet::sheet(&mut s, sound);
                at
            }
            Event::Found { step } => {
                sounds::found(&mut self.score(out), at, step);
                at
            }
        }
    }

    /// Where a tier's `landing` at `at` lands on the clock
    /// ([`tiers::lands`]): with the slam, a rung's on the next beat, a
    /// phrase's on a bar's first beat with room for it.
    fn lands(&self, at: u64, landing: Landing) -> u64 {
        self.clock
            .land(at, tiers::lands(landing, self.clock.beat_len()))
    }

    /// A chime of `tier` (none: under the first) from `at`, on the next
    /// beat; where its hit lands.
    fn chime(&mut self, out: &mut impl Commands, at: u64, tier: Option<Tier>) -> u64 {
        let hit = self.clock.land(at, When::Beat);
        sounds::chime(&mut self.score(out), at, hit, tier);
        hit
    }

    /// The streak gone at `at`: the payout to come and the glitter called
    /// off, what is held let go.
    fn lose(&mut self, out: &mut impl Commands, at: u64) {
        let mut s = self.score(out);
        s.send(Command::Cancel { tag: Some(PAYOUT) });
        s.send(Command::Cancel { tag: Some(STREAK) });
        sounds::let_go(&mut s, at);
        self.rung = None;
        self.riser = false;
    }

    /// A word's bubble popped on `at`, `streak` in a row with it, at `pan`
    /// and `size` big; the frame its pop (or the rung it reached) lands on.
    fn word(
        &mut self,
        out: &mut impl Commands,
        at: u64,
        streak: u32,
        (pan, size): (f32, f32),
    ) -> u64 {
        if streak == 0 {
            return at;
        }
        // Only the streak lost lets go of the rung reached ([`Event::Lost`]):
        // a word a step lower (told late, or a word rescued after those past
        // it) climbs on from there.
        let degree = ladder::degree(streak, &layers::rungs());
        let reached = rung(streak);
        let played = at_rung(reached);
        let rungs_up = (share(reached), played.contains(Streak::Glint));
        sounds::pop(&mut self.score(out), at, degree, rungs_up, pan, size);
        let second = f64::from(self.rate);
        if played.contains(Streak::Clinks) && at >= self.quiet_until {
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
        match reached {
            Some(tier) if reached > self.rung => lands = self.rung_up(out, at, tier),
            _ => {
                // The word before a rung a riser leads into: the riser,
                // until the rung comes.
                let next = (Tier::ALL.into_iter()).find(|t| t.spec().streak == streak + 1);
                let rises = next.is_some_and(|t| at_rung(Some(t)).contains(Streak::Riser));
                if rises && !self.riser {
                    sounds::riser(&mut self.score(out), at);
                    self.riser = true;
                }
                if played.contains(Streak::Glitter) {
                    let first = self.clock.next_step(at + 1, 4);
                    let sixteenth = (self.clock.beat_len() / 4.0) as u64;
                    let steps = [first, first + sixteenth];
                    sounds::glitter_after(&mut self.score(out), steps);
                }
            }
        }
        if played.contains(Streak::Shimmer) {
            // Started on the rung's hit; each word after holds it on.
            sounds::shimmer(&mut self.score(out), lands);
        }
        lands
    }

    /// `tier`'s rung reached on `at`, above the last: its stinger, and its
    /// payout where it has one; the frame its hit lands on.
    fn rung_up(&mut self, out: &mut impl Commands, at: u64, tier: Tier) -> u64 {
        self.rung = Some(tier);
        let hit = self.lands(at, Landing::Rung);
        if self.riser {
            sounds::riser_lands(&mut self.score(out), hit);
            self.riser = false;
        }
        let played = at_rung(Some(tier));
        let end = sounds::stinger(&mut self.score(out), at, hit, share(Some(tier)), played);
        self.quiet_until = self.quiet_until.max(end);
        if played.contains(Streak::Call) {
            self.announce(out, at, hit, Called::Times(tier));
        }
        if played.contains(Streak::Payout) {
            self.payout(out, end);
        }
        hit
    }

    /// The top rung's payout from `at` (its stinger's end), on a downbeat
    /// with room for it, its lead-in where there is room for that too.
    pub fn payout(&mut self, out: &mut impl Commands, at: u64) {
        let bar = self.clock.bar();
        let beat = self.clock.beat_len();
        let length = sounds::payout_beats(bar) * beat;
        let d = self.clock.land(at, When::Downbeat { length });
        let lead = (PAYOUT_LEAD * beat) as u64;
        let room = d >= at + lead && self.clock.clear(d - lead, lead);
        sounds::payout(&mut self.score(out), at, d, bar, room);
    }

    /// The announcer calling `called` out from `now`, its first vowel on
    /// `at`: as the game does where the slam lands (a phrase's [`Event::Phrase`]
    /// and a rung's [`Event::Word`] at the rows that call), softer where it
    /// would be sung over a line; a word it has none for, nothing.
    pub fn announce(&mut self, out: &mut impl Commands, now: u64, at: u64, called: Called) {
        let Some(word) = called.word() else {
            return;
        };
        let span = call::span(word, self.clock.beat_len(), f64::from(self.rate));
        let gain = if self.clock.clear(at, span) {
            1.0
        } else {
            call::DUCK
        };
        call::sing(&mut self.score(out), now, at, word, gain);
    }

    /// The score the sounds are written into, sending to `out`: its notes
    /// on their own frames, on the theme's patches, in the key, timed by
    /// the clock (a note written into it played as the conductor plays its
    /// own: the Synth tab's keys).
    pub(crate) fn score<'a>(&'a mut self, out: &'a mut dyn Commands) -> Score<'a> {
        Score {
            out,
            dropped: &mut self.dropped,
            theme: self.theme,
            key: self.key,
            beat: self.clock.beat_len(),
            bar: self.clock.bar(),
            rate: f64::from(self.rate),
            seed: &mut self.seed,
            origin: Origin::NONE,
        }
    }
}

/// A synth's tempo as it was told last, kept by whoever plays on it (the
/// one rule it is told by, however many planners play there: the game's
/// conductor and the Synth tab's, [`crate::sfx::Sfx`]). A synth starts at
/// the free clock's tempo ([`Conductor::synth`]), and so does this.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tempo {
    told: f64,
}

impl Default for Tempo {
    fn default() -> Tempo {
        Tempo {
            told: jodel_rhythm::grid::FREE_TEMPO,
        }
    }
}

impl Tempo {
    /// `c` timing its notes by `clock` from now on, and the synth told its
    /// tempo ([`Tempo::tell`]).
    pub fn time(&mut self, c: &mut Conductor, out: &mut impl Commands, clock: Clock) {
        c.time_by(clock);
        self.tell(out, clock.tempo());
    }

    /// The synth told the tempo `bpm` when it is more than half a beat a
    /// minute off what it was told last (which it is then, taken or not:
    /// a command lost is counted by where it went).
    pub fn tell(&mut self, out: &mut impl Commands, bpm: f64) {
        if (bpm - self.told).abs() > 0.5 {
            self.told = bpm;
            out.send(Command::Tempo { bpm: bpm as f32 });
        }
    }
}
