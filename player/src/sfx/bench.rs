//! The Synth tab's workbench ([`crate::core::synth_tab`]): every sound of
//! the game, listed by its moment ([`SOUNDS`], [`moments`]), each played
//! as the game plays it, by a conductor of the bench's own on the game's
//! synth ([`Bench`]: the game's conductor is never touched): a word's pop,
//! a run of words climbing to the first rung, a rung reached (its stinger,
//! its call and its payout where it has them), the rescue shot, a streak
//! lost, a phrase's chime, the payout and the jackpot alone, the
//! announcer's calls, the diploma's count-up, gems, points, rungs and its
//! sheet's own sounds, and a result of a song's preparation found; and the
//! keys of the tab's keyboard, each note held on its own tag until its key
//! is let go of. Which atoms a sound plays is found by writing it down
//! once ([`Sound::atoms`]): what it sends the synth, its patches named.

use std::ops::Range;
use std::sync::LazyLock;

use jodel_rhythm::key::Key;
use jodel_synth::{Command, Commands, Cue};

use super::conductor::{self, tags, Atom, Called, Clock, Conductor, Diploma, Event, Note, Sheet};
use crate::theme::Theme;
use crate::tiers::Tier;

/// A moment of the game, as the tab groups its sounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Moment {
    /// A word sung: its pop, the rescue shot, a streak lost.
    Words,
    /// A rung of the multiplier reached.
    Rungs,
    /// A phrase rated: its tier's chime.
    Chimes,
    /// The big wins: the payout, the jackpot.
    Wins,
    /// The announcer's calls.
    Announcer,
    /// The diploma's count-up, gems, points and rungs.
    Diploma,
    /// The diploma's sheet: its paper, pen, stamps and tunes.
    Sheet,
    /// A song got ready: a result found.
    Preparation,
}

impl Moment {
    /// Its name, as the tab heads its sounds.
    pub fn name(self) -> &'static str {
        match self {
            Moment::Words => "Words",
            Moment::Rungs => "Rungs",
            Moment::Chimes => "Chimes",
            Moment::Wins => "Wins",
            Moment::Announcer => "Announcer",
            Moment::Diploma => "Diploma",
            Moment::Sheet => "Diploma sheet",
            Moment::Preparation => "Getting ready",
        }
    }
}

/// What a sound is to the conductor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum What {
    /// A game event, as the game tells it.
    Event(Event),
    /// The word that reaches a tier's rung, from none: its pop, the rung's
    /// stinger, its call and its payout where it has them.
    Rung(Tier),
    /// Words in a row up to the first rung, a quarter of a second apart.
    Run,
    /// The top rung's payout alone.
    Payout,
    /// The announcer calling out.
    Call(Called),
    /// The diploma's points counting up.
    CountUp,
}

/// A sound of the game, as the tab lists it.
#[derive(Clone, Debug, PartialEq)]
pub struct Sound {
    /// Its moment.
    pub moment: Moment,
    /// Its name.
    pub name: String,
    /// What it is to the conductor.
    pub what: What,
    /// The tier whose layers it plays, if a tier's row decides them.
    pub tier: Option<Tier>,
    /// The layers that tier switches on for it, by name.
    pub layers: Vec<&'static str>,
    /// The atoms it plays, in the order it first plays them.
    pub atoms: Vec<Atom>,
}

/// Every sound, by moment.
pub static SOUNDS: LazyLock<Vec<Sound>> = LazyLock::new(catalogue);

/// Each moment with its sounds' places in [`SOUNDS`], in their order.
pub fn moments() -> impl Iterator<Item = (Moment, Range<usize>)> {
    let sounds = &*SOUNDS;
    let mut from = 0;
    std::iter::from_fn(move || {
        let moment = sounds.get(from)?.moment;
        let to = (sounds[from..].iter())
            .position(|s| s.moment != moment)
            .map_or(sounds.len(), |n| from + n);
        let span = from..to;
        from = to;
        Some((moment, span))
    })
}

/// The sounds with their atoms, written down by a conductor of their own.
fn catalogue() -> Vec<Sound> {
    let mut sounds = Vec::new();
    let mut add = |moment, name: String, what, tier: Option<Tier>, layers| {
        sounds.push(Sound {
            moment,
            name,
            what,
            tier,
            layers,
            atoms: atoms(what),
        });
    };
    let word = |streak, size| {
        What::Event(Event::Word {
            streak,
            pan: 0.0,
            size,
        })
    };
    add(
        Moment::Words,
        "A word".into(),
        word(1, 0.3),
        None,
        Vec::new(),
    );
    add(
        Moment::Words,
        "A word held long".into(),
        word(3, 1.0),
        None,
        Vec::new(),
    );
    add(
        Moment::Words,
        "A run to the first rung".into(),
        What::Run,
        Some(Tier::ALL[0]),
        streak_layers(Tier::ALL[0]),
    );
    add(
        Moment::Words,
        "The rescue shot".into(),
        What::Event(Event::Zap { pan: 0.0 }),
        None,
        Vec::new(),
    );
    for tier in [Tier::ALL[0], Tier::TOP] {
        add(
            Moment::Words,
            format!("Streak lost from {}", tier.name()),
            What::Event(Event::Lost(tier)),
            Some(tier),
            streak_layers(tier),
        );
    }
    for tier in Tier::ALL {
        let name = format!("{} ×{}", tier.name(), tier.multiplier());
        add(
            Moment::Rungs,
            name,
            What::Rung(tier),
            Some(tier),
            streak_layers(tier),
        );
    }
    for tier in Tier::ALL {
        add(
            Moment::Chimes,
            tier.name().into(),
            What::Event(Event::Phrase(tier)),
            Some(tier),
            chime_layers(tier),
        );
    }
    add(
        Moment::Wins,
        "The payout".into(),
        What::Payout,
        None,
        Vec::new(),
    );
    add(
        Moment::Wins,
        "The jackpot".into(),
        What::Event(Event::Jackpot),
        None,
        Vec::new(),
    );
    for tier in Tier::ALL {
        for (called, name) in [
            (Called::Name(tier), format!("“{}!”", tier.name())),
            (
                Called::Times(tier),
                format!("“Times {}!”", tier.multiplier()),
            ),
        ] {
            add(
                Moment::Announcer,
                name,
                What::Call(called),
                None,
                Vec::new(),
            );
        }
    }
    add(
        Moment::Diploma,
        "The count-up".into(),
        What::CountUp,
        None,
        Vec::new(),
    );
    let diploma = |d| What::Event(Event::Diploma(d));
    let tiers = std::iter::once(None).chain(Tier::ALL.map(Some));
    for tier in tiers.clone() {
        let name = tier.map_or("A gem", Tier::name);
        let gem = Diploma::Gem { tier, pan: 0.0 };
        add(
            Moment::Diploma,
            format!("Gem: {name}"),
            diploma(gem),
            tier,
            Vec::new(),
        );
    }
    for tier in tiers {
        let name = tier.map_or("none", Tier::name);
        let layers = tier.map_or_else(Vec::new, chime_layers);
        add(
            Moment::Diploma,
            format!("Points: {name}"),
            diploma(Diploma::Points(tier)),
            tier,
            layers,
        );
    }
    for tier in Tier::ALL {
        add(
            Moment::Diploma,
            format!("Best streak: {}", tier.name()),
            diploma(Diploma::Rung(tier)),
            Some(tier),
            streak_layers(tier),
        );
    }
    for sheet in Sheet::ALL {
        let sound = Diploma::Sheet {
            sound: sheet,
            gain: 1.0,
            pan: 0.0,
        };
        add(
            Moment::Sheet,
            sheet.title().into(),
            diploma(sound),
            None,
            Vec::new(),
        );
    }
    add(
        Moment::Preparation,
        "A result found".into(),
        What::Event(Event::Found { step: 0 }),
        None,
        Vec::new(),
    );
    sounds
}

/// The layers a tier's rung plays, by name.
fn streak_layers(tier: Tier) -> Vec<&'static str> {
    conductor::at_rung(Some(tier))
        .iter()
        .map(|l| l.name())
        .collect()
}

/// The layers a tier's chime plays, by name.
fn chime_layers(tier: Tier) -> Vec<&'static str> {
    (tier.spec().layers.chime.iter())
        .map(|l| l.name())
        .collect()
}

/// The gap between the words of a run, and between the count-up's ticks,
/// seconds.
const RUN_GAP: f64 = 0.25;
const TICK_GAP: f64 = 0.06;
/// The count-up's ticks.
const TICKS: u32 = 16;

/// How long a key held on the keyboard holds its note at most, seconds
/// (it is let go of with the key).
const KEY_HOLD: f64 = 10.0;

/// How a key of the keyboard is held: by the pointer, or a computer key
/// at its place on the keyboard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Held {
    /// The pointer.
    Pointer,
    /// The computer key at this place (0 up).
    Key(u8),
}

impl Held {
    /// Its notes' tag: its own ([`tags`]).
    fn tag(self) -> jodel_synth::Tag {
        match self {
            Held::Pointer => tags::POINTER,
            Held::Key(k) => tags::computer_key(k),
        }
    }
}

/// The workbench's player: a conductor of its own on the game's synth,
/// planning on the desk's frames, made anew for each sound (so a rung or a
/// run of words plays from none), on the patches of the theme it is
/// played in. The synth's tempo is kept by whoever plays the game's
/// conductor and this one on it ([`crate::sfx::Sfx`]): the bench's clock
/// only times its notes ([`Conductor::time_by`]).
pub struct Bench {
    conductor: Conductor,
}

impl Bench {
    /// One at `rate` Hz.
    pub fn new(rate: u32) -> Bench {
        Bench {
            conductor: Conductor::planner(rate, &crate::theme::CLASSIC),
        }
    }

    /// The rate it plans at, Hz.
    pub fn rate(&self) -> u32 {
        self.conductor.rate()
    }

    /// `what` from `at` in `theme`'s patches, in `key` (none: unpitched)
    /// on a steady beat at `bpm`, sent to `out`.
    pub fn play(
        &mut self,
        out: &mut impl Commands,
        at: u64,
        what: What,
        (theme, key, bpm): (&Theme, Option<&Key>, f64),
    ) {
        let rate = self.rate();
        let c = &mut self.conductor;
        *c = Conductor::planner(rate, theme);
        c.set_key(key, 0);
        c.time_by(Clock::steady(rate, bpm, at));
        play(c, out, at, what);
    }

    /// `note` on `atom`'s patch in `theme` from `at`, held by `held` until
    /// it lets go ([`Bench::up`], at most [`KEY_HOLD`]).
    pub fn down(
        &mut self,
        out: &mut impl Commands,
        (at, theme): (u64, &Theme),
        (atom, note): (Atom, f32),
        held: Held,
    ) {
        let hold = (KEY_HOLD * f64::from(self.rate())) as u64;
        let c = &mut self.conductor;
        c.set_theme(theme);
        let n = Note::new(at, atom, note).vel(0.85).hold(hold);
        c.score(out).play(n.tag(held.tag()));
    }

    /// `held`'s note let go of on `at`, alone (whatever patch and theme it
    /// was pressed on).
    pub fn up(&mut self, out: &mut impl Commands, at: u64, held: Held) {
        self.conductor.score(out).release(at, held.tag());
    }
}

/// `what` played from `at` on the conductor's frames, sent to `out`; its
/// clock and key as the caller set them, its streak as a song's start.
fn play(c: &mut Conductor, out: &mut impl Commands, at: u64, what: What) {
    let second = f64::from(c.rate());
    let after = |k: u32, gap: f64| at + (f64::from(k) * gap * second) as u64;
    match what {
        What::Event(event) => {
            c.event(out, at, event);
        }
        What::Rung(tier) => {
            let word = Event::Word {
                streak: tier.spec().streak,
                pan: 0.0,
                size: 0.5,
            };
            c.event(out, at, word);
        }
        What::Run => {
            for k in 1..=Tier::ALL[0].spec().streak {
                let pan = if k % 2 == 0 { 0.3 } else { -0.3 };
                let word = Event::Word {
                    streak: k,
                    pan,
                    size: 0.3,
                };
                c.event(out, after(k - 1, RUN_GAP), word);
            }
        }
        What::Payout => c.payout(out, at),
        What::Call(called) => c.announce(out, at, at, called),
        What::CountUp => {
            for k in 0..TICKS {
                let tick = Event::Diploma(Diploma::Tick { k, of: TICKS });
                c.event(out, after(k, TICK_GAP), tick);
            }
        }
    }
}

/// The atoms `what` plays (in A minor at 120 beats a minute), in the order
/// it first plays them.
fn atoms(what: What) -> Vec<Atom> {
    let mut bench = Bench::new(48_000);
    let mut out: Vec<Command> = Vec::new();
    let key = Key::on(9, true);
    bench.play(
        &mut out,
        0,
        what,
        (&crate::theme::CLASSIC, Some(&key), 120.0),
    );
    let mut atoms = Vec::new();
    for command in out {
        if let Command::At {
            cue: Cue::Note { patch, .. },
            ..
        } = command
        {
            if let Some((_, a)) = conductor::of(patch).filter(|(_, a)| !atoms.contains(a)) {
                atoms.push(a);
            }
        }
    }
    atoms
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every sound plays at least one atom, and is named apart from the
    /// others of its moment; every moment has sounds, in the order the
    /// tab lists them.
    #[test]
    fn every_sound_plays_and_is_named_apart() {
        let sounds = &*SOUNDS;
        assert!(sounds.len() > 40, "{}", sounds.len());
        for (i, s) in sounds.iter().enumerate() {
            assert!(!s.atoms.is_empty(), "{}: plays nothing", s.name);
            assert!(
                sounds[i + 1..]
                    .iter()
                    .all(|t| t.moment != s.moment || t.name != s.name),
                "{} twice",
                s.name
            );
        }
        let mut moments: Vec<Moment> = sounds.iter().map(|s| s.moment).collect();
        moments.dedup();
        assert_eq!(
            moments,
            [
                Moment::Words,
                Moment::Rungs,
                Moment::Chimes,
                Moment::Wins,
                Moment::Announcer,
                Moment::Diploma,
                Moment::Sheet,
                Moment::Preparation
            ]
        );
        let top = sounds.iter().find(|s| s.what == What::Rung(Tier::TOP));
        assert!(
            top.is_some_and(|s| s.atoms.contains(&Atom::Singer)),
            "its call"
        );
        // Each moment's sounds together, in the order listed.
        let spans: Vec<Moment> = super::moments().map(|(m, _)| m).collect();
        assert_eq!(spans, moments);
        assert!(super::moments().all(|(m, span)| sounds[span].iter().all(|s| s.moment == m)));
    }

    /// A key of the keyboard holds its note on the atom's patch until it
    /// is let go of, alone: the pointer and a computer key on the very
    /// same pitch are two notes, and the one let go of goes, the other
    /// held on; the theme's patch plays; a key pressed in one theme and let
    /// go of after the tab compares another is let go of all the same.
    #[test]
    fn a_key_holds_its_note_until_it_is_let_go_of() {
        use crate::theme::{CLASSIC, SPOOKY};
        const RATE: u32 = 48_000;
        let (_, mut synth) = Conductor::new(RATE, &CLASSIC);
        let mut bench = Bench::new(RATE);
        bench.down(&mut synth, (0, &SPOOKY), (Atom::Choir, 60.0), Held::Pointer);
        bench.down(&mut synth, (0, &SPOOKY), (Atom::Choir, 60.0), Held::Key(4));
        bench.down(&mut synth, (0, &SPOOKY), (Atom::Choir, 64.0), Held::Key(7));
        let mut out = vec![0.0f32; 2 * 24_000];
        synth.render_to(&mut out);
        let held = |synth: &jodel_synth::Synth| {
            let mut notes: Vec<f32> = synth.voices().filter(|v| v.held).map(|v| v.note).collect();
            notes.sort_by(f32::total_cmp);
            notes
        };
        assert_eq!(held(&synth), [60.0, 60.0, 64.0]);
        let choir = conductor::id(conductor::place(&SPOOKY), Atom::Choir);
        assert!(synth.voices().all(|v| v.patch == choir));
        let at = synth.frame();
        bench.up(&mut synth, at, Held::Pointer);
        synth.render_to(&mut out);
        assert_eq!(held(&synth), [60.0, 64.0]);
        let at = synth.frame();
        bench.up(&mut synth, at, Held::Key(4));
        synth.render_to(&mut out);
        assert_eq!(held(&synth), [64.0]);
        // The theme switched, a sound of it played, the key still held let
        // go of: nothing held.
        let at = synth.frame();
        bench.play(
            &mut synth,
            at,
            What::Event(Event::Zap { pan: 0.0 }),
            (&CLASSIC, None, 120.0),
        );
        synth.render_to(&mut out);
        let at = synth.frame();
        bench.up(&mut synth, at, Held::Key(7));
        synth.render_to(&mut out);
        assert!(held(&synth).is_empty(), "{:?}", held(&synth));
        assert_eq!(synth.dropped(), 0);
    }
}
