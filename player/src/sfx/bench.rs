//! The Synth tab's workbench ([`crate::core::synth_tab`]): every sound of
//! the game, listed by its moment ([`SOUNDS`]), each played through the
//! conductor as the game plays it ([`play`]): a word's pop, a run of words
//! climbing to the first rung, a rung reached (its stinger, its call and
//! its payout where it has them), the rescue shot, a streak lost, a
//! phrase's chime, the payout and the jackpot alone, the announcer's
//! calls, and the diploma's count-up, gems, points, rungs and its sheet's
//! own sounds. Which atoms a sound plays is found by writing it down once
//! ([`Sound::atoms`]): what it sends the synth, its patches named.

use std::sync::LazyLock;

use jodel_synth::{Command, Commands};

use super::conductor::{Atom, Called, Clock, Conductor, Diploma, Event, Sheet};
use crate::core::singing::Side;
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
            if called.sung() {
                add(
                    Moment::Announcer,
                    name,
                    What::Call(called),
                    None,
                    Vec::new(),
                );
            }
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
            sheet_name(sheet),
            diploma(sound),
            None,
            Vec::new(),
        );
    }
    sounds
}

/// A sheet's sound by name, as the tab lists it.
fn sheet_name(sheet: Sheet) -> String {
    use super::conductor::Entry;
    let entry = |e| match e {
        Entry::Words => "the words",
        Entry::Streak => "the streak",
    };
    match sheet {
        Sheet::Unroll => "Unrolled".into(),
        Sheet::Star(Side::Left) => "A star, left".into(),
        Sheet::Star(Side::Right) => "A star, right".into(),
        Sheet::Title => "The title".into(),
        Sheet::Written(e) => format!("Written: {}", entry(e)),
        Sheet::Scribbled(e) => format!("Scribbled: {}", entry(e)),
        Sheet::Sticker => "A sticker".into(),
        Sheet::Cheer => "A cheer".into(),
        Sheet::Whole => "Not from the start".into(),
        Sheet::Snooze => "Nothing scored".into(),
        Sheet::Medal(m) => format!("Medal: {}", m.name()),
        Sheet::Stamp => "The stamp".into(),
        Sheet::NewBest => "A new best".into(),
        Sheet::Signature => "Signed".into(),
        Sheet::Date => "Dated".into(),
        Sheet::Tap => "A tap of the pen".into(),
    }
}

/// The layers a tier's rung plays, by name.
fn streak_layers(tier: Tier) -> Vec<&'static str> {
    use super::conductor::Streak;
    (tier.spec().layers.streak.iter())
        .map(|l| match l {
            Streak::Glint => "glint",
            Streak::Octave => "octave",
            Streak::Register => "register",
            Streak::Clinks => "clinks",
            Streak::Reels => "reels",
            Streak::Brass => "brass",
            Streak::Riser => "riser",
            Streak::Shimmer => "shimmer",
            Streak::Glitter => "glitter",
            Streak::Payout => "payout",
            Streak::Call => "call",
        })
        .collect()
}

/// The layers a tier's chime plays, by name.
fn chime_layers(tier: Tier) -> Vec<&'static str> {
    use super::conductor::Chime;
    (tier.spec().layers.chime.iter())
        .map(|l| match l {
            Chime::Plucks => "plucks",
            Chime::BrassHit => "brass hit",
            Chime::TaDaa => "ta-daa",
            Chime::Choir => "choir",
            Chime::Call => "call",
        })
        .collect()
}

/// The gap between the words of a run, and between the count-up's ticks,
/// seconds.
const RUN_GAP: f64 = 0.25;
const TICK_GAP: f64 = 0.06;
/// The count-up's ticks.
const TICKS: u32 = 16;

/// `what` played from `at` on the conductor's frames, sent to `out`; its
/// clock and key as the caller set them.
pub fn play(c: &mut Conductor, out: &mut impl Commands, at: u64, what: What) {
    let second = f64::from(c.rate());
    let after = |k: u32, gap: f64| at + (f64::from(k) * gap * second) as u64;
    match what {
        What::Event(event) => {
            if matches!(event, Event::Word { .. }) {
                c.start_over();
            }
            c.event(out, at, event);
        }
        What::Rung(tier) => {
            c.start_over();
            let word = Event::Word {
                streak: tier.spec().streak,
                pan: 0.0,
                size: 0.5,
            };
            c.event(out, at, word);
        }
        What::Run => {
            c.start_over();
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

/// What a sound sends, written down.
#[derive(Default)]
struct Written(Vec<Command>);

impl Commands for Written {
    fn send(&mut self, c: Command) -> bool {
        self.0.push(c);
        true
    }
}

/// The atoms `what` plays (in A minor at 120 beats a minute), in the order
/// it first plays them.
fn atoms(what: What) -> Vec<Atom> {
    const RATE: u32 = 48_000;
    let (mut c, _) = Conductor::planner(RATE, &crate::theme::CLASSIC);
    let mut out = Written::default();
    c.set_key(Some(&jodel_rhythm::key::Key::on(9, true)), 0);
    c.set_clock(&mut out, Clock::steady(RATE, 120.0, 0));
    play(&mut c, &mut out, 0, what);
    let mut atoms = Vec::new();
    for command in out.0 {
        if let Command::Note { patch, .. } = command {
            if let Some(a) = c.atom_of(patch).filter(|a| !atoms.contains(a)) {
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
                Moment::Sheet
            ]
        );
        let top = sounds.iter().find(|s| s.what == What::Rung(Tier::TOP));
        assert!(
            top.is_some_and(|s| s.atoms.contains(&Atom::Singer)),
            "its call"
        );
    }
}
