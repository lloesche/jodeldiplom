//! The phrase tiers, the only place one is defined: a row each, Good
//! first, the top (Diva) last ([`TIERS`]), and a tier named by its place
//! there ([`Tier`]). A phrase reaches the highest row whose bars its shine
//! clears ([`Tier::of`], with the scoring in [`crate::core::shine`]);
//! everything a tier looks and sounds like is read from its row: its
//! colour on the stage and the diploma, the meter's fill, how strongly the
//! singing glows while at it and what that lights (the streak's rainbow,
//! the lane, the backdrop), its mark on the song's progress, its sticker,
//! the layers of its sounds (the streak's at its rung, a phrase's chime:
//! [`TierSpec::layers`]). Each row is a rung of the multiplier too, named
//! after it: reached at so many words hit in a row ([`TierSpec::streak`]),
//! ×2 the first's, doubling a rung ([`crate::core::singing::rung`],
//! [`Tier::multiplier`]).
//!
//! A row may have to be earned in a row (Diva's: [`TierSpec::earned`]): a
//! phrase reaching it counts as it only right after a phrase that counted
//! at least the row below, else as that row ([`Tier::counted`]).
//!
//! A look held while the singing glows is a value a row: between two rows
//! (the glow easing from one to the other) it is the straight line between
//! their values ([`held`]), below the first from none.
//!
//! The rows are in order: each bar, streak and glow above the one before,
//! and each row looks and sounds other than the one before (a test holds
//! them to it).

use crate::gui::fx::{Effects, Fx, Use};
use crate::sfx::conductor::{Chime, Layers, Streak};

/// How a tier fills the meter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Paint {
    /// A colour.
    Rgb([f32; 3]),
    /// The rainbow, running along it.
    Rainbow,
}

/// How a tier's sticker looks on the diploma.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sticker {
    /// Whether it glints now and then while the screen twinkles.
    pub glints: bool,
    /// Whether its colour runs a little through the rainbow.
    pub prismatic: bool,
    /// How strong its halo is (0: none); with one it also lights what is
    /// round it, twinkles and bursts as it lands.
    pub halo: f32,
}

/// A tier: what it takes and how it shows.
#[derive(Debug)]
pub struct TierSpec {
    /// Its name, as shown (its sounds are named after it too, in lower
    /// case).
    pub name: &'static str,
    /// The shine it takes, where the original reaches 1.
    pub bar: f64,
    /// The share of the original's pitch it takes too, if it asks one.
    pub pitch_bar: Option<f64>,
    /// Whether it must be earned in a row: a phrase clearing its bars counts
    /// as it only right after a phrase that counted at least the row below
    /// (else as that row: [`Tier::counted`]); never the first row's.
    pub earned: bool,
    /// The words hit in a row its rung of the multiplier takes (the rung,
    /// named after it: [`Tier::multiplier`]).
    pub streak: u32,
    /// Its colour (sRGB): its name's badge, its stretch on the song's
    /// progress and its mark there, its sticker on the diploma.
    pub color: [f32; 3],
    /// How the meter fills from its mark up.
    pub fill: Paint,
    /// How strongly the singing glows while at it, 0 to 1 (the top's 1):
    /// the streak's rainbow, the ring of the time left and the text in
    /// rainbow ([`crate::gui::rainbow::Glow`]), and where its held looks
    /// are its own ([`held`]).
    pub glow: f32,
    /// How many bands the rainbow fans out from the playhead (a whole
    /// number).
    pub bands: f32,
    /// How much the rainbow lights the whole lane, 0 to 1 (held).
    pub lane_light: f32,
    /// The effects it shows: while the singing is at it, when a phrase
    /// reaches it, at its rung of the multiplier.
    pub fx: Effects,
    /// How brightly the backdrop's subject is lit, 0 to 1 (held).
    pub backdrop_glow: f32,
    /// How strong the spotlight rays from behind it are, 0 to 1 (held).
    pub backdrop_rays: f32,
    /// How much harder the subject moves with the beat (its push, its
    /// rim's light, a phrase landing's flare), 0 to 1 over none (held:
    /// [`crate::gui::backdrop::pulse`]).
    pub backdrop_beat: f32,
    /// How large its phrases' marks on the song's progress are, in the
    /// bar's heights.
    pub mark: f32,
    /// How its sticker looks on the diploma.
    pub sticker: Sticker,
    /// What its sounds are made of: the streak's at its rung, a phrase's
    /// chime reaching it (the game's synth plays them:
    /// [`crate::sfx::conductor`]).
    pub layers: Layers,
}

/// The tiers, Good first.
pub static TIERS: [TierSpec; 5] = [
    TierSpec {
        name: "Good",
        bar: 0.60,
        pitch_bar: None,
        earned: false,
        streak: 5,
        color: [0.35, 0.92, 0.62],
        fill: Paint::Rainbow,
        glow: 0.25,
        bands: 10.0,
        lane_light: 0.0,
        fx: Effects {
            held: &[Use::of(Fx::Rainbow, 1.0)],
            on_phrase: &[Use::of(Fx::StepUp, 1.0)],
            ..Effects::NONE
        },
        backdrop_glow: 0.0,
        backdrop_rays: 0.0,
        backdrop_beat: 0.1,
        mark: 1.8,
        sticker: Sticker {
            glints: false,
            prismatic: false,
            halo: 0.0,
        },
        layers: Layers {
            streak: &[],
            chime: &[],
        },
    },
    TierSpec {
        name: "Great",
        bar: 0.72,
        pitch_bar: None,
        earned: false,
        streak: 10,
        color: [1.0, 0.78, 0.22],
        fill: Paint::Rgb([1.0, 0.72, 0.16]),
        glow: 0.5,
        bands: 14.0,
        lane_light: 0.0,
        fx: Effects {
            held: &[Use::of(Fx::Rainbow, 1.0), Use::of(Fx::Glow, 1.0)],
            on_phrase: &[Use::of(Fx::StepUp, 1.0)],
            ..Effects::NONE
        },
        backdrop_glow: 0.35,
        backdrop_rays: 0.0,
        backdrop_beat: 0.25,
        mark: 1.8,
        sticker: Sticker {
            glints: false,
            prismatic: false,
            halo: 0.0,
        },
        layers: Layers {
            streak: &[Streak::Glint, Streak::Octave],
            chime: &[Chime::Plucks],
        },
    },
    TierSpec {
        name: "Awesome",
        bar: 0.80,
        pitch_bar: None,
        earned: false,
        streak: 20,
        color: [1.0, 0.56, 0.3],
        fill: Paint::Rgb([1.0, 0.5, 0.22]),
        glow: 0.625,
        bands: 16.0,
        lane_light: 0.5,
        fx: Effects {
            held: &[
                Use::of(Fx::Rainbow, 1.0),
                Use::of(Fx::Glow, 1.0),
                Use::of(Fx::EdgeFire, 0.3),
            ],
            on_phrase: &[Use::of(Fx::StepUp, 1.0)],
            on_rung: &[Use::of(Fx::Fountain, 1.0)],
            at_rung: &[Use::of(Fx::Flick, 1.0)],
        },
        backdrop_glow: 0.5,
        backdrop_rays: 0.08,
        backdrop_beat: 0.45,
        mark: 1.8,
        sticker: Sticker {
            glints: true,
            prismatic: false,
            halo: 0.0,
        },
        layers: Layers {
            streak: &[
                Streak::Glint,
                Streak::Octave,
                Streak::Register,
                Streak::Clinks,
                Streak::Reels,
            ],
            chime: &[Chime::Plucks, Chime::BrassHit],
        },
    },
    TierSpec {
        name: "Amazing",
        bar: 0.87,
        pitch_bar: None,
        earned: false,
        streak: 35,
        color: [1.0, 0.35, 0.6],
        fill: Paint::Rgb([1.0, 0.3, 0.58]),
        glow: 0.75,
        bands: 18.0,
        lane_light: 1.0,
        fx: Effects {
            held: &[
                Use::of(Fx::Rainbow, 1.0),
                Use::of(Fx::Glow, 1.0),
                Use::of(Fx::Flames, 1.0),
                Use::of(Fx::EdgeFire, 0.6),
                Use::of(Fx::LyricFire, 0.3),
            ],
            on_phrase: &[Use::of(Fx::StepUp, 1.0)],
            on_rung: &[Use::of(Fx::Fountain, 1.0)],
            at_rung: &[Use::of(Fx::Flick, 1.0), Use::of(Fx::Bulbs, 0.6)],
        },
        backdrop_glow: 0.65,
        backdrop_rays: 0.5,
        backdrop_beat: 0.7,
        mark: 1.8,
        sticker: Sticker {
            glints: true,
            prismatic: false,
            halo: 0.0,
        },
        layers: Layers {
            streak: &[
                Streak::Glint,
                Streak::Octave,
                Streak::Register,
                Streak::Clinks,
                Streak::Reels,
                Streak::Brass,
                Streak::Riser,
                Streak::Shimmer,
            ],
            chime: &[Chime::Plucks, Chime::TaDaa],
        },
    },
    TierSpec {
        name: "Diva",
        bar: 0.94,
        pitch_bar: Some(0.92),
        earned: true,
        streak: 55,
        color: [0.96, 0.93, 1.0],
        fill: Paint::Rgb([1.0, 0.95, 0.85]),
        glow: 1.0,
        bands: 22.0,
        lane_light: 1.0,
        fx: Effects {
            held: &[
                Use::of(Fx::Rainbow, 1.0),
                Use::of(Fx::Glow, 1.0),
                Use::of(Fx::Flames, 1.0),
                Use::of(Fx::Angels, 1.0),
                Use::of(Fx::EdgeFire, 1.0),
                Use::of(Fx::LyricFire, 0.6),
            ],
            on_phrase: &[
                Use::of(Fx::StepUp, 1.0),
                Use::of(Fx::Jackpot, 1.0),
                Use::of(Fx::Slam, 1.0),
            ],
            on_rung: &[
                Use::of(Fx::Fountain, 1.0),
                Use::of(Fx::Rain, 1.0),
                Use::of(Fx::Slam, 1.0),
            ],
            at_rung: &[Use::of(Fx::Flick, 1.0), Use::of(Fx::Bulbs, 1.0)],
        },
        backdrop_glow: 1.0,
        backdrop_rays: 1.0,
        backdrop_beat: 1.0,
        mark: 2.4,
        sticker: Sticker {
            glints: true,
            prismatic: true,
            halo: 0.5,
        },
        layers: Layers {
            streak: &[
                Streak::Glint,
                Streak::Octave,
                Streak::Register,
                Streak::Clinks,
                Streak::Reels,
                Streak::Brass,
                Streak::Riser,
                Streak::Shimmer,
                Streak::Glitter,
                Streak::Payout,
            ],
            chime: &[Chime::Plucks, Chime::TaDaa, Chime::Choir],
        },
    },
];

/// A held look at glow `k`: the straight line through none at 0 and each
/// row's `value` at its glow, the top's from there on. At a row's glow it
/// is that row's value exactly.
pub fn held(k: f32, value: impl Fn(&TierSpec) -> f32) -> f32 {
    held_through(k, TIERS.iter().map(|row| (row.glow, value(row))))
}

/// A held value at glow `k` through `points` (each a row's glow and its
/// value there, the glows rising): the straight line through none at 0
/// and each point, the last's from there on; at a point's glow its value
/// exactly.
pub fn held_through(k: f32, points: impl IntoIterator<Item = (f32, f32)>) -> f32 {
    let mut a = (0.0, 0.0);
    for b in points {
        if k < b.0 {
            return a.1 + (b.1 - a.1) * ((k - a.0) / (b.0 - a.0));
        }
        a = b;
    }
    a.1
}

/// A value a row, Good first, as a WGSL `array<f32, TIERS>` (the shaders
/// have `TIERS`, how many there are: [`crate::gui::rainbow::stage_wgsl`]).
pub fn wgsl_array(values: impl IntoIterator<Item = f32>) -> String {
    let values: Vec<String> = values.into_iter().map(|v| format!("{v:?}")).collect();
    format!("array<f32, TIERS>({})", values.join(", "))
}

/// How well a phrase shone: a row of [`TIERS`] by its place there, Good
/// first, the higher the better.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tier(u8);

impl Tier {
    /// How many there are.
    pub const COUNT: usize = TIERS.len();
    /// All, Good first.
    pub const ALL: [Tier; Tier::COUNT] = {
        let mut all = [Tier(0); Tier::COUNT];
        let mut i = 0;
        while i < Tier::COUNT {
            all[i] = Tier(i as u8);
            i += 1;
        }
        all
    };
    /// The top: Diva.
    pub const TOP: Tier = Tier(Tier::COUNT as u8 - 1);

    /// Its row.
    pub fn spec(self) -> &'static TierSpec {
        &TIERS[self.index()]
    }

    /// Its place among the tiers, Good's 0.
    pub fn index(self) -> usize {
        usize::from(self.0)
    }

    /// The shine it takes, where the original reaches 1.
    pub fn bar(self) -> f64 {
        self.spec().bar
    }

    /// Its name, as shown.
    pub fn name(self) -> &'static str {
        self.spec().name
    }

    /// The tier named `name` (as shown), for tests and examples that mean
    /// one tier in particular.
    ///
    /// # Panics
    ///
    /// If there is none of that name.
    pub fn named(name: &str) -> Tier {
        Tier::ALL
            .into_iter()
            .find(|t| t.name() == name)
            .unwrap_or_else(|| panic!("no tier named {name:?}"))
    }

    /// The multiplier at its rung: ×2 the first's, doubling a rung.
    pub fn multiplier(self) -> u32 {
        2 << self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row's looks and sounds, to tell two rows apart by: its numbers,
    /// and how strong it holds and fires each effect.
    fn looks(t: &TierSpec) -> Vec<f64> {
        let s = t.sticker;
        let effects = (t.fx.lists().into_iter())
            .flat_map(|list| Fx::ALL.map(|fx| f64::from(crate::gui::fx::strength(list, fx))));
        [
            t.glow,
            t.bands,
            t.lane_light,
            t.backdrop_glow,
            t.backdrop_rays,
            t.backdrop_beat,
            t.mark,
            s.halo,
        ]
        .map(f64::from)
        .into_iter()
        .chain([
            f64::from(u8::from(s.glints)),
            f64::from(u8::from(s.prismatic)),
        ])
        .chain(effects)
        .collect()
    }

    /// Every row in order: each bar, streak, glow, band count, backdrop
    /// glow and beat above the one before, the held looks never less, each
    /// row looking other than the one before (a row that copies another's
    /// looks fails here), and sounding more: its streak's layers all the
    /// row before's and more, its chime's the row before's (a "ta-daa" in
    /// place of a short brass hit) and other; the top's glow 1, its rung
    /// paying out, its phrase's chime a choir; each named apart, a mark on
    /// the meter, and its rung reached at its streak, the one before a word
    /// short of it.
    #[test]
    fn every_row_is_in_order() {
        for w in TIERS.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            let pair = format!("{} {}", a.name, b.name);
            assert!(b.bar > a.bar, "{pair}");
            assert!(b.streak > a.streak, "{pair}");
            assert!(b.glow > a.glow, "{pair}");
            assert!(b.bands > a.bands, "{pair}");
            assert!(b.backdrop_glow > a.backdrop_glow, "{pair}");
            assert!(b.backdrop_beat > a.backdrop_beat, "{pair}");
            for (x, y) in looks(a).iter().zip(looks(b)) {
                assert!(y >= *x, "{pair}");
            }
            assert_ne!(looks(a), looks(b), "{pair}");
            let (la, lb) = (a.layers, b.layers);
            assert!(la.streak.iter().all(|l| lb.streak.contains(l)), "{pair}");
            assert!(lb.streak.len() > la.streak.len(), "{pair}");
            let kept = |l: &Chime| {
                lb.chime.contains(l) || (*l == Chime::BrassHit && lb.chime.contains(&Chime::TaDaa))
            };
            assert!(la.chime.iter().all(kept), "{pair}");
            assert_ne!(la.chime, lb.chime, "{pair}");
        }
        assert_eq!(Tier::TOP.spec().glow, 1.0);
        assert!(TIERS[0].glow > 0.0 && TIERS[0].streak > 0);
        let top = Tier::TOP.spec().layers;
        assert!(top.streak.contains(&Streak::Payout) && top.chime.contains(&Chime::Choir));
        assert_eq!(crate::core::singing::rung(0), None);
        let marks = crate::gui::glory::MARKS;
        for t in Tier::ALL {
            assert_eq!(Tier::named(t.name()), t);
            assert_eq!(t.spec().bands.fract(), 0.0, "{}", t.name());
            let streak = t.spec().streak;
            let below = t.index().checked_sub(1).map(|i| Tier::ALL[i]);
            assert_eq!(crate::core::singing::rung(streak), Some(t));
            assert_eq!(crate::core::singing::rung(streak - 1), below);
            assert!(marks[t.index()] > 0.0 && marks[t.index()] <= 1.0);
        }
        assert!(marks.windows(2).all(|w| w[1] > w[0]));
        assert_eq!(marks[Tier::TOP.index()], 1.0);
    }

    /// A held look: none at 0, each row's own value at its glow (exactly),
    /// the straight line between, the top's past it.
    #[test]
    fn a_held_look_runs_through_the_rows() {
        let value = |t: &TierSpec| t.backdrop_glow + 1.0;
        assert_eq!(held(0.0, value), 0.0);
        for t in &TIERS {
            assert_eq!(held(t.glow, value).to_bits(), value(t).to_bits());
        }
        let (a, b) = (&TIERS[0], &TIERS[1]);
        let mid = held((a.glow + b.glow) / 2.0, value);
        assert!((mid - (value(a) + value(b)) / 2.0).abs() < 1e-6);
        let half = held(a.glow / 2.0, value);
        assert!((half - value(a) / 2.0).abs() < 1e-6);
        assert_eq!(held(2.0, value), value(Tier::TOP.spec()));
    }

    /// A row earned in a row is above the first (it has a row below to be
    /// earned after and to count as instead).
    #[test]
    fn every_earned_row_has_a_row_below() {
        for (i, t) in TIERS.iter().enumerate() {
            assert!(!t.earned || i > 0, "{}", t.name);
        }
    }

    #[test]
    #[should_panic(expected = "no tier named")]
    fn a_tier_misnamed_is_none_at_all() {
        Tier::named("Fabulous");
    }
}
